"""
max adversarial training framework
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import time

import torch
import torch.optim as optim
import numpy as np

from core.attack.max import Max
from core.defense.advdet_gmm import EXP_OVER_FLOW
from core.defense.principled_adv_training import PrincipledAdvTraining
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.principled_adv_training')
logger.addHandler(ErrorHandler)


class MaxAdvTraining(object):
    """max adversarial training

    Parameters
    -------
    @param model, Object,  a model to be protected, e.g., MalwareDetector
    @attack_model: Object, adversary's model for generating adversarial malware on the feature space
    """

    def __init__(self, model, attack_model=None, attack_param=None):
        self.model = model
        if attack_model is not None:
            assert isinstance(attack_model, Max)
            if 'is_attacker' in attack_model.__dict__.keys():
                assert not attack_model.is_attacker
        self.attack_model = attack_model
        self.attack_param = attack_param

        self.name = self.model.name
        self.model_save_path = path.join(config.get('experiments', 'm_adv_training') + '_' + self.name,
                                         'model.pth')
        self.model.model_save_path = self.model_save_path

    def fit(self, train_data_producer, validation_data_producer=None, epochs=5, adv_epochs=20,
            beta_a=0.001,
            lambda_lower_bound=1e-3,
            lambda_upper_bound=1e3,
            granularity=1,
            lr=0.005,
            weight_decay=5e-4, verbose=True):
        """
        Applying adversarial train to enhance the malware detector.

        Parameters
        -------
        @param train_data_producer: Object, an dataloader object for producing a batch of training data
        @param validation_data_producer: Object, an dataloader object for producing validation dataset
        @param epochs: Integer, epochs for adversarial training
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
        # logger.info("Normal training is starting...")
        # self.model.fit(train_data_producer,
        #                validation_data_producer,
        #                epochs=epochs,
        #                lr=lr,
        #                weight_decay=weight_decay)
        # # get threshold tau
        # self.model.get_threshold(validation_data_producer)
        # logger.info(f"The threshold is {self.model.tau:.3f}.")
        #
        # optimizer = optim.Adam(self.model.customize_param(weight_decay), lr=lr, weight_decay=weight_decay)
        # total_time = 0.
        # nbatches = len(train_data_producer)
        # lambda_space = np.logspace(np.log10(lambda_lower_bound),
        #                            np.log10(lambda_upper_bound),
        #                            num=int(np.log10(lambda_upper_bound / lambda_lower_bound) // granularity) + 1)
        # logger.info("Max adversarial training is starting ...")
        # for i in range(adv_epochs):
        #     losses, accuracies = [], []
        #     for ith_batch, res in enumerate(train_data_producer):
        #         x_batch, adj_batch, y_batch, _1 = res
        #         x_batch, adj_batch, y_batch = utils.to_tensor(x_batch, adj_batch, y_batch, self.model.device)
        #         batch_size = x_batch.shape[0]
        #
        #         # perturb malware feature vectors
        #         mal_x_batch, mal_adj_batch, mal_y_batch, null_flag = PrincipledAdvTraining.get_mal_data(x_batch,
        #                                                                                                 adj_batch,
        #                                                                                                 y_batch)
        #         if null_flag:
        #             continue
        #         mal_batch_size = mal_x_batch.shape[0]
        #         start_time = time.time()
        #         # the attack perturbs feature vectors using various hyper-parameter lambda, aiming to obtain
        #         # adversarial examples as much as possible
        #         lambda_ = np.random.choice(lambda_space)
        #         self.model.eval()
        #         pertb_mal_x = self.attack_model.perturb(self.model, mal_x_batch, mal_adj_batch, mal_y_batch,
        #                                                 steps_of_max=self.attack_param['steps'],
        #                                                 min_lambda_=lambda_,
        #                                                 # when lambda is small, we cannot get effective attacks
        #                                                 max_lambda_=lambda_upper_bound,
        #                                                 verbose=self.attack_param['verbose']
        #                                                 )
        #         total_time += time.time() - start_time
        #         x_batch = torch.vstack([x_batch, pertb_mal_x])
        #         if adj_batch is not None:
        #             adj_batch = torch.vstack([adj_batch, mal_adj_batch])
        #
        #         start_time = time.time()
        #         self.model.train()
        #         optimizer.zero_grad()
        #         hidden, logits = self.model.forward(x_batch, adj_batch)
        #         loss_train = self.model.customize_loss(logits[:batch_size],
        #                                                y_batch,
        #                                                hidden[:batch_size],
        #                                                ith_batch)
        #         # appending adversarial training loss
        #         # loss_train += self.model.customize_loss(logits[batch_size: batch_size + mal_batch_size],
        #         #                                         mal_y_batch,
        #         #                                         hidden[batch_size: batch_size + mal_batch_size],
        #         #                                         ith_batch)
        #         # appending adversarial training loss
        #         loss_train += beta_a * torch.mean(
        #             torch.log(self.model.forward_g(hidden[batch_size: batch_size + mal_batch_size]) + EXP_OVER_FLOW))
        #
        #         loss_train.backward()
        #         optimizer.step()
        #         total_time += time.time() - start_time
        #
        #         acc_train = (logits.argmax(1) == torch.cat([y_batch, mal_y_batch])).sum().item()
        #         acc_train /= x_batch.size()[0]
        #         mins, secs = int(total_time / 60), int(total_time % 60)
        #         losses.append(loss_train.item())
        #         accuracies.append(acc_train)
        #         if verbose:
        #             logger.info(
        #                 f'Mini batch: {i * nbatches + ith_batch + 1}/{adv_epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
        #             logger.info(
        #                 f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}')
        #     # get threshold tau
        #     self.model.get_threshold(validation_data_producer)
        #     logger.info(f"The threshold is {self.model.tau:.3f}.")
        #     if not path.exists(self.model_save_path):
        #         utils.mkdir(path.dirname(self.model_save_path))
        #     torch.save({'model_state_dict': self.model.state_dict(),
        #                 'epoch': adv_epochs,
        #                 'optimizer_state_dict': optimizer.state_dict()
        #                 },
        #                self.model_save_path)
        #     # save the inter-model periodically for model selection (: todo)
        #     torch.save({'model_state_dict': self.model.state_dict(),
        #                 'epoch': adv_epochs,
        #                 'optimizer_state_dict': optimizer.state_dict()
        #                 },
        #                self.model_save_path + str(i // 5 + 1))
        #     if verbose:
        #         logger.info(
        #             f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
        #         logger.info(f'The threshold is {self.model.tau}.')

        # pick a model
        ckpt_id = set([i // 5 + 1 for i in range(adv_epochs)])
        best_acc_val = 0.
        best_epoch = 0
        for id in ckpt_id:
            ckpt = torch.load(self.model_save_path + str(id))
            self.model.load_state_dict(ckpt['model_state_dict'])
            res = []
            for x_val_batch, adj_val_batch, y_val_batch, _1 in validation_data_producer:
                mal_x_batch, mal_adj_batch, mal_y_batch, null_flag = PrincipledAdvTraining.get_mal_data(x_val_batch,
                                                                                                        adj_val_batch,
                                                                                                        y_val_batch)
                if null_flag:
                    continue
                with torch.no_grad():
                    pertb_mal_x = self.attack_model.perturb(self.model, mal_x_batch, mal_adj_batch, mal_y_batch,
                                                            steps_of_max=self.attack_param['steps'],
                                                            min_lambda_=lambda_lower_bound,
                                                            max_lambda_=lambda_upper_bound,
                                                            verbose=self.attack_param['verbose']
                                                            )
                    y_cent_batch, x_density_batch = self.model.inference_batch_wise(pertb_mal_x,
                                                                                    mal_adj_batch,
                                                                                    mal_y_batch,
                                                                                    use_indicator=True)
                    y_pred = np.argmax(y_cent_batch, axis=-1)
                    indicator_flag = self.model.indicator(x_density_batch, y_pred)
                    res.append((~indicator_flag) | ((y_pred == 1.) & indicator_flag))
            res = np.concatenate(res)
            acc_val = np.sum(res).astype(np.float) / res.shape[0]
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                best_epoch = id * 5 - 1
                torch.save({'model_state_dict': self.model.state_dict(),
                            'epoch': ckpt['epoch'],
                            'optimizer_state_dict': ckpt['optimizer_state_dict']
                            },
                           self.model_save_path)

        if verbose:
            logger.info(f"Val: model select at epoch {best_epoch} with validation accuracy {best_acc_val*100}% under attack.")

    def load(self):
        ckpt = torch.load(self.model_save_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
