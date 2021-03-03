"""
A adversarial training framework
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

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

    def __init__(self, model, attack_model):
        self.model = model
        self.attack_model = attack_model

        self.name = self.model.name + '_prin_adv'
        self.model_save_path = path.join(config.get('experiments', 'prip_adv_training') + '_' + self.name,
                                         'model.pth')
        self.model.model_save_path = self.model_save_path

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=5e-4, verbose=True):
        """
        Train the malware detector, pick the best model according to the cross-entropy loss on validation set

        Parameters
        -------
        @param train_data_producer: Object, an iterator for producing a batch of training data
        @param validation_data_producer: Object, an iterator for producing validation dataset
        @param epochs: Integer, epochs
        @param lr: Float, learning rate for Adam optimizer
        @param weight_decay: Float, penalty factor, default value 5e-4 in graph attention layer
        @param verbose: Boolean, whether to show verbose logs
        """
        optimizer = optim.Adam(self.model.param_customizing(weight_decay), lr=lr)
        best_avg_acc = 0.
        best_epoch = 0
        total_time = 0.
        nbatchs = len(train_data_producer)
        for i in range(epochs):
            losses, accuracies = [], []
            for idx_batch, res in enumerate(train_data_producer):
                x_batch, adj, y_batch = res

                # perturb malware feature vectors
                x_batch, adj_batch, y_batch = utils.to_tensor(x_batch, adj, y_batch, self.model.device)
                mal_x_batch, mal_adj_batch, mal_y_batch = self.get_mal_data(x_batch, adj_batch, y_batch)
                start_time = time.time()
                adv_x_batch = self.attack_model.perturb(self.model, mal_x_batch, mal_adj_batch, mal_y_batch)
                total_time += time.time() - start_time
                batch_size = x_batch.shape[0]
                x_batch = torch.vstack([x_batch, adv_x_batch])
                if adj is not None:
                    adj_batch = torch.vstack([adj_batch, mal_adj_batch])

                # start training
                start_time = time.time()
                self.model.train()
                optimizer.zero_grad()
                latent_rpst, logits = self.model.forward(x_batch, adj_batch)
                loss_train = self.model.customize_loss(logits[:batch_size], y_batch, latent_rpst[:batch_size], idx_batch)
                loss_train += F.cross_entropy(logits[batch_size:], mal_y_batch)
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
                        f'Mini batch: {i * nbatchs + idx_batch + 1}/{epochs * nbatchs} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}')

            self.model.eval()
            avg_acc_val = []
            for res in validation_data_producer:
                x_val, adj_val, y_val = res
                x_val, adj_val, y_val = utils.to_tensor(x_val, adj_val, y_val, self.model.device)
                mal_x_val, mal_adj_val, mal_y_val = self.get_mal_data(x_val, adj_val, y_val)
                adv_x_val = self.attack_model.perturb(self.model, mal_x_val, mal_adj_val, mal_y_val)
                x_val = torch.cat([x_val, adv_x_val])
                if adj_val is not None:
                    adj_val = torch.vstack([adj_val, mal_adj_val])

                _, logits = self.model.forward(x_val, adj_val)
                acc_val = (logits.argmax(1) == torch.cat([y_val, mal_y_val])).sum().item()
                acc_val /= x_val.size()[0]
                avg_acc_val.append(acc_val)
                avg_acc_val = np.mean(avg_acc_val)

            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                if not path.exists(self.model_save_path):
                    utils.mkdir(path.dirname(self.model_save_path))
                torch.save(self.model.state_dict(), self.model_save_path)
                if verbose:
                    print(f'Model saved at path: {self.model_save_path}')

            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    @staticmethod
    def get_mal_data(x_batch, adj_batch, y_batch):
        mal_x_batch = x_batch[y_batch == 1]
        mal_y_batch = y_batch[y_batch == 1]
        mal_adj_batch = None
        if adj_batch is not None:
            mal_adj_batch = torch.stack([adj for i, adj in enumerate(adj_batch) if y_batch[i] == 1], dim=0)
        return mal_x_batch, mal_adj_batch, mal_y_batch
