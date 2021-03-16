"""
A adversarial training framework
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from core.defense.advdet_gmm import EXP_OVER_FLOW
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.principled_adv_training')
logger.addHandler(ErrorHandler)


class PrincipledAdvTraining(object):
    """a framework of principled adversarial training for defending against adversarial malware

    Parameters
    ------------------
    @param model, Object,  a model to be protected, e.g., MalwareDetector
    @attack_model: Object, adversary's model for generating adversarial malware on the feature space
    """

    def __init__(self, model, attack_model=None, attack_param=None):
        self.model = model
        self.attack_model = attack_model
        self.attack_param = attack_param

        self.name = self.model.name
        self.model_save_path = path.join(config.get('experiments', 'prip_adv_training') + '_' + self.name,
                                         'model.pth')
        self.model.model_save_path = self.model_save_path

    def fit(self, train_data_producer, validation_data_producer, epochs=100, adv_epochs=20, epsilon=1, lr=0.005,
            weight_decay=5e-4, verbose=True):
        """
        Train the malware detector, pick the best model according to the cross-entropy loss on validation set

        Parameters
        -------
        @param train_data_producer: Object, an iterator for producing a batch of training data
        @param validation_data_producer: Object, an iterator for producing validation dataset
        @param epochs: Integer, epochs
        @param adv_epochs: Integer, epochs for adversarial training
        @param epsilon: Float, a small number of perturbations
        @param lr: Float, learning rate for Adam optimizer
        @param weight_decay: Float, penalty factor, default value 5e-4 in graph attention layer
        @param verbose: Boolean, whether to show verbose logs
        """
        assert epsilon <= self.attack_param['m']
        # normal training
        logger.info("Training is starting...")
        self.model.fit(train_data_producer,
                       validation_data_producer,
                       epochs=epochs,
                       lr=lr,
                       weight_decay=weight_decay)
        # get tau
        self.model.get_threshold(validation_data_producer)
        logger.info(f"The threshold is {self.model.tau:.3f}.")

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_time = 0.
        nbatchs = len(train_data_producer)

        # self.model.sample_weights[1] /= 2.  # owing to augmenting the training set using malware
        logger.info("Adversarial training is starting ...")
        for i in range(adv_epochs):
            losses, accuracies = [], []
            for idx_batch, res in enumerate(train_data_producer):
                x_batch, adj, y_batch = res
                x_batch, adj_batch, y_batch = utils.to_tensor(x_batch, adj, y_batch, self.model.device)
                batch_size = x_batch.shape[0]
                # perturb malware feature vectors
                mal_x_batch, mal_adj_batch, mal_y_batch, null_flag = self.get_mal_data(x_batch, adj_batch, y_batch)
                mal_batch_size = mal_x_batch.shape[0]
                if null_flag:
                    continue
                start_time = time.time()
                pert_x_batch = self.attack_model.perturb(self.model, mal_x_batch, mal_adj_batch, mal_y_batch,
                                                         self.attack_param['m'],
                                                         1e-3,
                                                         1e3,
                                                         self.attack_param['verbose']
                                                         )
                # continue
                # pert_x_batch_ext = self.attack_model.perturb(self.model, pert_x_batch, mal_adj_batch, mal_y_batch,
                #                                              epsilon,
                #                                              1e-3,
                #                                              1e3,
                #                                              self.attack_param['verbose']
                #                                              )
                total_time += time.time() - start_time
                # perturbations = torch.sum(torch.abs(adv_x_batch - mal_x_batch), dim=(1, 2))
                # adv_ce_flag = (perturbations <= epsilon)
                x_batch = torch.vstack([x_batch, pert_x_batch])
                if adj is not None:
                    adj_batch = torch.vstack([adj_batch, mal_adj_batch, mal_adj_batch])

                # start training
                start_time = time.time()
                self.model.train()
                optimizer.zero_grad()
                latent_rpst, logits = self.model.forward(x_batch, adj_batch)
                loss_train = self.model.customize_loss(logits[:batch_size],
                                                       y_batch,
                                                       latent_rpst[:batch_size],
                                                       idx_batch)
                loss_train += F.cross_entropy(logits[batch_size:batch_size + mal_batch_size], mal_y_batch)
                loss_train += self.model.beta * torch.mean(
                    torch.log(self.model.forward_g(latent_rpst[batch_size: batch_size + mal_batch_size]) + EXP_OVER_FLOW))
                print('adv:', self.model.forward_g(latent_rpst[batch_size: batch_size + mal_batch_size]))

                loss_train.backward()
                optimizer.step()
                total_time += time.time() - start_time
                acc_train = (logits.argmax(1) == torch.cat([y_batch, mal_y_batch])).sum().item()
                acc_train /= x_batch.size()[0]
                mins, secs = int(total_time / 60), int(total_time % 60)
                losses.append(loss_train.item())
                accuracies.append(acc_train)
                if verbose:
                    print(
                        f'Mini batch: {i * nbatchs + idx_batch + 1}/{adv_epochs * nbatchs} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}')

            if not path.exists(self.model_save_path):
                utils.mkdir(path.dirname(self.model_save_path))
            if (i + 1) % 10 == 0:
                torch.save(self.model.state_dict(), path.join(path.dirname(self.model_save_path), f'model{i + 1}.pth'))
            self.model.get_threshold(validation_data_producer)
            torch.save(self.model.state_dict(), self.model_save_path)
            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'The threshold is {self.model.tau}.'
                )

    @staticmethod
    def get_mal_data(x_batch, adj_batch, y_batch):
        mal_x_batch = x_batch[y_batch == 1]
        mal_y_batch = y_batch[y_batch == 1]
        mal_adj_batch = None
        if adj_batch is not None:
            mal_adj_batch = torch.stack([adj for i, adj in enumerate(adj_batch) if y_batch[i] == 1], dim=0)
        null_flag = len(mal_x_batch) <= 0
        return mal_x_batch, mal_adj_batch, mal_y_batch, null_flag
