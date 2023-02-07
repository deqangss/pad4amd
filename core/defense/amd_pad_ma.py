"""
max adversarial training framework
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import random
import time

import torch
import torch.optim as optim
import numpy as np

from core.attack.max import Max
from core.attack.stepwise_max import StepwiseMax
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.max_adv_training')
logger.addHandler(ErrorHandler)


class AMalwareDetectionPAD(object):
    """adversarial training incorporating ``max'' or step-wise ``max'' attack

    Parameters
    -------
    @param model, Object,  a model to be protected, e.g., MalwareDetector
    @attack_model: Object, adversary's model for generating adversarial malware on the feature space
    """

    def __init__(self, model, attack=None, attack_param=None):
        self.model = model
        assert hasattr(self.model, 'forward_g')
        if attack is not None:
            assert isinstance(attack, (Max, StepwiseMax))
            if 'is_attacker' in attack.__dict__.keys():
                assert not attack.is_attacker
        self.attack = attack
        self.attack_param = attack_param

        self.name = self.model.name
        self.model_save_path = path.join(config.get('experiments', 'amd_pad_ma') + '_' + self.name,
                                         'model.pth')
        self.model.model_save_path = self.model_save_path

    def fit(self, train_data_producer, validation_data_producer=None, adv_epochs=50,
            beta_1=0.1,
            beta_2=1,
            lmda_lower_bound=1e-3,
            lmda_upper_bound=1e3,
            use_continuous_pert=True,
            lr=0.001,
            under_sampling_ratio=1.,
            weight_decay=5e-0,
            verbose=True):
        """
        Applying adversarial train to enhance the malware detector.

        Parameters
        -------
        @param train_data_producer: Object, an dataloader object for producing a batch of training data
        @param validation_data_producer: Object, an dataloader object for producing validation dataset
        @param adv_epochs: Integer, epochs for adversarial training
        @param beta_1: Float, penalty factor for adversarial loss
        @param beta_2: Float, penalty factor for adversarial loss
        @param lmda_lower_bound: Float, lower boundary of penalty factor
        @param lmda_upper_bound: Float, upper boundary of penalty factor
        @param use_continuous_pert: Boolean, whether use continuous perturbations or not
        @param lr: Float, learning rate of Adam optimizer
        @param under_sampling_ratio: [0,1], under-sampling a portion of malware examples for adversarial training
        @param weight_decay: Float, penalty factor, default value 5e-4 in Graph ATtention layer (GAT)
        @param verbose: Boolean, whether to show verbose info
        """
        constraint = utils.NonnegWeightConstraint()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_time = 0.
        nbatches = len(train_data_producer)
        lmda_space = np.logspace(np.log10(lmda_lower_bound),
                                 np.log10(lmda_upper_bound),
                                 num=int(np.log10(lmda_upper_bound / lmda_lower_bound)) + 1)
        logger.info("Max adversarial training is starting ...")
        best_acc_val = 0.
        acc_val_adv_be = 0.
        best_epoch = 0
        for i in range(adv_epochs):
            random.seed(0)
            losses, accuracies = [], []
            for idx_batch, (x_batch, y_batch) in enumerate(train_data_producer):
                x_batch, y_batch = utils.to_tensor(x_batch.double(), y_batch.long(), self.model.device)
                batch_size = x_batch.shape[0]
                # make data
                # 1. add pepper and salt noises for adversary detector
                x_batch_noises = torch.clamp(x_batch + utils.psn(x_batch, np.random.uniform(0, 0.5)), min=0., max=1.)
                x_batch_ = torch.cat([x_batch, x_batch_noises], dim=0)
                y_batch_ = torch.cat([torch.zeros(batch_size, ), torch.ones(batch_size, )]).long().to(
                    self.model.device)
                idx = torch.randperm(y_batch_.shape[0])
                x_batch_ = x_batch_[idx]
                y_batch_ = y_batch_[idx]
                # 2. data for classifier
                mal_x_batch, ben_x_batch, mal_y_batch, ben_y_batch, null_flag = \
                    utils.get_mal_ben_data(x_batch, y_batch)
                if 0. < under_sampling_ratio < 1.:
                    n_mal = mal_x_batch.shape[0]
                    n_mal_sampling = int(under_sampling_ratio * n_mal) if int(under_sampling_ratio * n_mal) > 1 else 1
                    idx_sampling = random.sample(range(n_mal), n_mal_sampling)
                    mal_x_batch, mal_y_batch = mal_x_batch[idx_sampling], mal_y_batch[idx_sampling]
                if null_flag:
                    continue
                start_time = time.time()
                # the attack perturbs feature vectors using various hyper-parameter lambda, aiming to obtain
                # adversarial examples as much as possible
                self.model.eval()
                pertb_mal_x = self.attack.perturb(self.model, mal_x_batch, mal_y_batch,
                                                  min_lambda_=np.random.choice(lmda_space),
                                                  # when lambda is small, we cannot get effective attacks
                                                  max_lambda_=lmda_upper_bound,
                                                  **self.attack_param
                                                  )
                disc_pertb_mal_x_ = utils.round_x(pertb_mal_x, 0.5)
                total_time += time.time() - start_time
                x_batch = torch.cat([x_batch, disc_pertb_mal_x_], dim=0)
                y_batch = torch.cat([y_batch, mal_y_batch])
                if use_continuous_pert:
                    filter_flag = torch.amax(torch.abs(pertb_mal_x - mal_x_batch), dim=-1) <= 1e-6
                    pertb_mal_x = pertb_mal_x[~filter_flag]
                    orgin_mal_x = mal_x_batch[~filter_flag]
                    x_batch_ = torch.cat([x_batch_, orgin_mal_x, pertb_mal_x], dim=0)
                    n_pertb_mal = pertb_mal_x.shape[0]
                else:
                    filter_flag = torch.sum(torch.abs(disc_pertb_mal_x_ - mal_x_batch), dim=-1) == 0
                    disc_pertb_mal_x_ = disc_pertb_mal_x_[~filter_flag]
                    orgin_mal_x = mal_x_batch[~filter_flag]
                    x_batch_ = torch.cat([x_batch_, orgin_mal_x, disc_pertb_mal_x_], dim=0)
                    n_pertb_mal = disc_pertb_mal_x_.shape[0]
                y_batch_ = torch.cat([y_batch_, torch.zeros((n_pertb_mal * 2,), ).to(
                    self.model.device)]).double()
                y_batch_[-n_pertb_mal:] = 1.
                start_time = time.time()
                self.model.train()
                optimizer.zero_grad()
                logits_f = self.model.forward_f(x_batch)
                logits_g = self.model.forward_g(x_batch_)
                loss_train = self.model.customize_loss(logits_f[:batch_size],
                                                       y_batch[:batch_size],
                                                       logits_g[:2 * batch_size],
                                                       y_batch_[:2 * batch_size])
                loss_train += self.model.customize_loss(logits_f[batch_size:],
                                                        y_batch[batch_size:],
                                                        logits_g[2 * batch_size:],
                                                        y_batch_[2 * batch_size:],
                                                        beta_1=beta_1,
                                                        beta_2=beta_2
                                                        )

                loss_train.backward()
                optimizer.step()
                # clamp
                for name, module in self.model.named_modules():
                    if 'non_neg_layer' in name:
                        module.apply(constraint)

                total_time += time.time() - start_time
                mins, secs = int(total_time / 60), int(total_time % 60)
                acc_f_train = (logits_f.argmax(1) == y_batch).sum().item()
                acc_f_train /= x_batch.size()[0]
                accuracies.append(acc_f_train)
                losses.append(loss_train.item())
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{adv_epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                if hasattr(self.model, 'forward_g'):
                    acc_g_train = ((torch.sigmoid(logits_g) >= 0.5) == y_batch_).sum().item()
                    acc_g_train /= x_batch_.size()[0]
                    accuracies.append(acc_g_train)
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_f_train * 100:.2f}% & {acc_g_train * 100:.2f}%.')
                else:
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_f_train * 100:.2f}%.')
            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')

            self.save_to_disk(self.model_save_path + '.tmp', i + 1, optimizer)
            # select model
            self.model.eval()
            self.attack.is_attacker = True
            res_val = []
            avg_acc_val = []
            for x_val, y_val in validation_data_producer:
                x_val, y_val = utils.to_tensor(x_val.double(), y_val.long(), self.model.device)
                logits_f = self.model.forward_f(x_val)
                acc_val = (logits_f.argmax(1) == y_val).sum().item()
                acc_val /= x_val.size()[0]
                avg_acc_val.append(acc_val)

                mal_x_batch, mal_y_batch, null_flag = utils.get_mal_data(x_val, y_val)
                if null_flag:
                    continue
                pertb_mal_x = self.attack.perturb(self.model, mal_x_batch, mal_y_batch,
                                                  min_lambda_=1e-5,
                                                  max_lambda_=1e5,
                                                  **self.attack_param
                                                  )
                y_cent_batch, x_density_batch = self.model.inference_batch_wise(pertb_mal_x)
                if hasattr(self.model, 'indicator'):
                    indicator_flag = self.model.indicator(x_density_batch)
                else:
                    indicator_flag = np.ones([x_density_batch.shape[0], ]).astype(np.bool)
                y_pred = np.argmax(y_cent_batch, axis=-1)
                res_val.append((~indicator_flag) | ((y_pred == 1.) & indicator_flag))
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
                self.save_to_disk(self.model_save_path)
            if verbose:
                logger.info(
                    f"\tVal accuracy {acc_val * 100:.4}% with accuracy {acc_val_adv * 100:.4}% under attack.")
                logger.info(
                    f"\tModel select at epoch {best_epoch} with validation accuracy {best_acc_val * 100:.4}% and accuracy {acc_val_adv_be * 100:.4}% under attack.")
            self.attack.is_attacker = False

    def load(self):
        assert path.exists(self.model_save_path), 'train model first'
        ckpt = torch.load(self.model_save_path)
        self.model.load_state_dict(ckpt['model'])

    def save_to_disk(self, save_path, epoch=None, optimizer=None):
        if not path.exists(path.dirname(save_path)):
            utils.mkdir(path.dirname(save_path))
        if epoch is not None and optimizer is not None:
            torch.save({'model': self.model.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()
                        },
                       save_path)
        else:
            torch.save({'model': self.model.state_dict()}, save_path)
