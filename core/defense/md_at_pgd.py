"""
adversarial training incorporating pgd linf attack
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import time

import torch
import torch.optim as optim
import numpy as np

from core.attack.pgd import PGD
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.adv_pgd_linf')
logger.addHandler(ErrorHandler)


class PGDAdvTraining(object):
    """adversarial training incorporating pgd attack

    Parameters
    -------
    @param model, Object,  a model to be protected, e.g., MalwareDetector
    @attack_model: Object, adversary's model for generating adversarial malware on the feature space
    """

    def __init__(self, model, attack=None, attack_param=None):
        self.model = model
        if attack is not None:
            assert isinstance(attack, PGD)
            if 'is_attacker' in attack.__dict__.keys():
                assert not attack.is_attacker
        self.attack = attack
        self.attack_param = attack_param

        self.name = self.model.name
        self.model_save_path = path.join(config.get('experiments', 'md_at_pgd') + '_' + self.name,
                                         'model.pth')
        self.model.model_save_path = self.model_save_path
        logger.info("Adversarial training incorporating the attack {}".format(type(self.attack).__name__))

    def fit(self, train_data_producer, validation_data_producer=None, epochs=5, adv_epochs=45,
            beta=0.001,
            lr=0.005,
            weight_decay=5e-0, verbose=True):
        """
        Applying adversarial train to enhance the malware detector.

        Parameters
        -------
        @param train_data_producer: Object, an dataloader object for producing a batch of training data
        @param validation_data_producer: Object, an dataloader object for producing validation dataset
        @param epochs: Integer, epochs for adversarial training
        @param adv_epochs: Integer, epochs for adversarial training
        @param beta: Float, penalty factor for adversarial loss
        @param lr: Float, learning rate of Adam optimizer
        @param weight_decay: Float, penalty factor, default value 5e-4
        @param verbose: Boolean, whether to show verbose info
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_time = 0.
        nbatches = len(train_data_producer)
        logger.info("adversarial training is starting ...")
        best_acc_val = 0.
        acc_val_adv_be = 0.
        best_epoch = 0
        for i in range(adv_epochs):
            losses, accuracies = [], []
            for idx_batch, (x_batch, y_batch) in enumerate(train_data_producer):
                x_batch, y_batch = utils.to_tensor(x_batch.double(), y_batch.long(), self.model.device)
                batch_size = x_batch.shape[0]
                # make data
                mal_x_batch, ben_x_batch, mal_y_batch, ben_y_batch, null_flag = \
                    utils.get_mal_ben_data(x_batch, y_batch)
                if null_flag:
                    continue
                start_time = time.time()
                self.model.eval()
                pertb_mal_x = self.attack.perturb(self.model, mal_x_batch, mal_y_batch,
                                                  **self.attack_param
                                                  )
                total_time += time.time() - start_time
                x_batch = torch.cat([ben_x_batch, pertb_mal_x], dim=0)
                y_batch = torch.cat([ben_y_batch, mal_y_batch])
                start_time = time.time()
                self.model.train()
                optimizer.zero_grad()
                logits = self.model.forward(x_batch)
                loss_train = self.model.customize_loss(logits,
                                                       y_batch)

                loss_train.backward()
                optimizer.step()

                total_time += time.time() - start_time
                mins, secs = int(total_time / 60), int(total_time % 60)
                acc_train = (logits.argmax(1) == y_batch).sum().item()
                acc_train /= x_batch.size()[0]
                accuracies.append(acc_train)
                losses.append(loss_train.item())
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{adv_epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}%.')
            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')

            # select model
            self.model.eval()
            # long-time to train (save the model temporally in case of interruption)
            self.save_to_disk(i + 1, optimizer, self.model_save_path + '.tmp')

            res_val = []
            avg_acc_val = []
            for x_val, y_val in validation_data_producer:
                x_val, y_val = utils.to_tensor(x_val.double(), y_val.long(), self.model.device)
                logits = self.model.forward(x_val)
                acc_val = (logits.argmax(1) == y_val).sum().item()
                acc_val /= x_val.size()[0]
                avg_acc_val.append(acc_val)

                mal_x_batch, mal_y_batch, null_flag = utils.get_mal_data(x_val, y_val)
                if null_flag:
                    continue
                pertb_mal_x = self.attack.perturb(self.model, mal_x_batch, mal_y_batch,
                                                  **self.attack_param
                                                  )
                y_cent_batch, x_density_batch = self.model.inference_batch_wise(pertb_mal_x)
                y_pred = np.argmax(y_cent_batch, axis=-1)
                res_val.append(y_pred == 1.)
            assert len(res_val) > 0
            res_val = np.concatenate(res_val)
            acc_val_adv = np.sum(res_val).astype(np.float) / res_val.shape[0]
            acc_val = (np.mean(avg_acc_val) + acc_val_adv) / 2.
            # Owing to we look for a new threshold after each epoch, this hinders the convergence of training.
            # We save the model's parameters at last several epochs as a well-trained model may be obtained.
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                acc_val_adv_be = acc_val_adv
                best_epoch = i + 1
                self.save_to_disk(best_epoch, optimizer, self.model_save_path)

            if verbose:
                logger.info(
                    f"\tVal accuracy {acc_val * 100:.4}% with accuracy {acc_val_adv * 100:.4}% under attack.")
                logger.info(
                    f"\tModel select at epoch {best_epoch} with validation accuracy {best_acc_val * 100:.4}% and accuracy {acc_val_adv_be * 100:.4}% under attack.")
                if hasattr(self.model, 'tau'):
                    logger.info(
                        f'The threshold is {self.model.tau}.'
                    )

    def load(self):
        assert path.exists(self.model_save_path), 'train model first'
        ckpt = torch.load(self.model_save_path)
        self.model.load_state_dict(ckpt['model'])

    def save_to_disk(self, epoch, optimizer, save_path=None):
        if not path.exists(save_path):
            utils.mkdir(path.dirname(save_path))
        torch.save({'model': self.model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()
                    },
                   save_path)
