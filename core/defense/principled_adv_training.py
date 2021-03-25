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
    """a framework of towards principled adversarial training

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
        self.model_save_path = path.join(config.get('experiments', 'p_adv_training') + '_' + self.name,
                                         'model.pth')
        self.model.model_save_path = self.model_save_path

    def fit(self, train_data_producer, validation_data_producer, epochs=100, adv_epochs=20,
            beta_a=0.001,
            lambda_lower_bound=1e-3,
            lambda_upper_bound=1e3,
            granularity=1,
            lr=0.005,
            weight_decay=5e-4, verbose=True):
        """
        Applying adversarial train to enhance the malware detector. Actually, we do not ensure this will
        produce a malware detector with principled adversarial training because we adjust the hyper-parameter
        lambda empirically.

        Parameters
        -------
        @param train_data_producer: Object, an dataloader object for producing a batch of training data
        @param validation_data_producer: Object, an dataloader object for producing validation dataset
        @param epochs: Integer, epochs for normal training (i.e., no perturbed examples are attended)
        @param adv_epochs: Integer, epochs for adversarial training
        @param beta_a: Float, penalty factor for adversarial loss
        @param lambda_lower_bound: Float, lower boundary of penalty factor
        @param lambda_upper_bound: Float, upper boundary of penalty factor
        @param granularity: Integer, 10^base exp-space between penalty factors
        @param lr: Float, learning rate of Adam optimizer
        @param weight_decay: Float, penalty factor, default value 5e-4 in Graph ATtention layer (GAT)
        @param verbose: Boolean, whether to show verbose info
        """
        # normal training is used for obtaining the initial indicator g
        logger.info("Normal training is starting...")
        self.model.fit(train_data_producer,
                       validation_data_producer,
                       epochs=epochs,
                       lr=lr,
                       weight_decay=weight_decay)
        # get threshold tau
        self.model.get_threshold(validation_data_producer)
        logger.info(f"The threshold is {self.model.tau:.3f}.")

        optimizer = optim.Adam(self.model.param_customizing(weight_decay), lr=lr, weight_decay=weight_decay)
        total_time = 0.
        nbatches = len(train_data_producer)
        lambda_space = np.logspace(np.log10(lambda_lower_bound),
                                   np.log10(lambda_upper_bound),
                                   num=int(np.log10(lambda_upper_bound / lambda_lower_bound)//granularity) + 1)
        logger.info("Adversarial training is starting ...")
        for i in range(adv_epochs):
            losses, accuracies = [], []
            for ith_batch, res in enumerate(train_data_producer):
                x_batch, adj_batch, y_batch, _1 = res
                x_batch, adj_batch, y_batch = utils.to_tensor(x_batch, adj_batch, y_batch, self.model.device)
                batch_size = x_batch.shape[0]

                # perturb malware feature vectors
                mal_x_batch, mal_adj_batch, mal_y_batch, null_flag = self.get_mal_data(x_batch, adj_batch, y_batch)
                mal_batch_size = mal_x_batch.shape[0]
                if null_flag:  # if no malware in this batch, exit straightly
                    continue

                start_time = time.time()
                # the attack perturbs feature vectors using various hyper-parameter lambda, aiming to obtain
                # adversarial examples as much as possible
                pertb_mal_x = self.attack_model.perturb(self.model, mal_x_batch, mal_adj_batch, mal_y_batch,
                                                        self.attack_param['m'],
                                                        lambda_=np.random.choice(lambda_space),
                                                        step_length=self.attack_param['step_length'],
                                                        verbose=self.attack_param['verbose']
                                                        )
                total_time += time.time() - start_time
                x_batch = torch.vstack([x_batch, pertb_mal_x])
                if adj_batch is not None:
                    adj_batch = torch.vstack([adj_batch, mal_adj_batch])

                start_time = time.time()
                self.model.train()
                optimizer.zero_grad()
                hidden, logits = self.model.forward(x_batch, adj_batch)
                loss_train = self.model.customize_loss(logits[:batch_size],
                                                       y_batch,
                                                       hidden[:batch_size],
                                                       ith_batch)
                # appending adversarial training loss
                loss_train += beta_a * torch.mean(
                    torch.log(self.model.forward_g(hidden[batch_size: batch_size + mal_batch_size]) + EXP_OVER_FLOW))

                loss_train.backward()
                optimizer.step()
                total_time += time.time() - start_time

                acc_train = (logits.argmax(1) == torch.cat([y_batch, mal_y_batch])).sum().item()
                acc_train /= x_batch.size()[0]
                mins, secs = int(total_time / 60), int(total_time % 60)
                losses.append(loss_train.item())
                accuracies.append(acc_train)
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + ith_batch + 1}/{adv_epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}')

            if not path.exists(self.model_save_path):
                utils.mkdir(path.dirname(self.model_save_path))
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
        """
        filter out malware feature vectors and adjacency matrix (if it is necessary)
        """
        mal_x_batch = x_batch[y_batch == 1]
        mal_y_batch = y_batch[y_batch == 1]
        mal_adj_batch = None
        if adj_batch is not None:
            mal_adj_batch = torch.stack([adj for i, adj in enumerate(adj_batch) if y_batch[i] == 1], dim=0)
        null_flag = len(mal_x_batch) <= 0
        return mal_x_batch, mal_adj_batch, mal_y_batch, null_flag
