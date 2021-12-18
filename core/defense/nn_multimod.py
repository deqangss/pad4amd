from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from tqdm import tqdm
import os.path as path
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from captum.attr import IntegratedGradients

import numpy as np

from core.defense.gcn import GCN
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.multimod')
logger.addHandler(ErrorHandler)


class MulModMalwareDetector(nn.Module):
    def __init__(self, input_dim_dnn, input_dim_gcn, n_classes, device='cpu', name='MULTIMOD', **kwargs):
        """
        Construct malware detector

        Parameters
        ----------
        @param vocab_size: Integer, the number of words in the vocabulary
        @param n_classes: Integer, the number of classes, n=2
        @param n_sample_times: Integer, the number of sampling times for predicting
        @param device: String, 'cpu' or 'cuda'
        @param name: String, model name
        """
        super(MulModMalwareDetector, self).__init__()

        self.input_dim_dnn = input_dim_dnn
        self.input_dim_gcn = input_dim_gcn
        self.n_classes = n_classes
        self.device = device
        self.name = name
        self.parse_args(**kwargs)

        # the name ``embedding_weight'' is not changable
        self.embedding_weight = nn.Parameter(torch.empty(size=(self.input_dim_gcn, self.embedding_dim)))
        nn.init.normal_(self.embedding_weight.data)  # default initialization method in torch

        self.gcn = GCN(self.embedding_dim,
                       self.hidden_units[0],
                       self.hidden_units[1],
                       dropout=self.dropout,
                       with_relu=self.with_relu,
                       with_bias=self.with_bias,
                       smooth=self.smooth,
                       alpha_=self.alpha_,
                       device=self.device
                       )

        self.dense_layers = []
        if 0 < len(self.dense_hidden_units) <= 1:
            self.dense_layers.append(nn.Linear(self.input_dim_dnn + self.hidden_units[1], self.dense_hidden_units[0]))
        elif len(self.dense_hidden_units) > 1:
            self.dense_layers.append(nn.Linear(self.input_dim_dnn, self.dense_hidden_units[0]))
        else:
            raise ValueError("Expect at least one hidden layer.")

        for i in range(len(self.dense_hidden_units[0:-1])):
            if i == len(self.dense_hidden_units) - 2:
                self.dense_layers.append(nn.Linear(self.dense_hidden_units[i] + self.hidden_units[1],
                                                   self.dense_hidden_units[i + 1]))
            else:
                self.dense_layers.append(nn.Linear(self.dense_hidden_units[i],
                                                   self.dense_hidden_units[i + 1]))
        self.dense_layers.append(nn.Linear(self.dense_hidden_units[-1], self.n_classes))
        # registration
        for idx_i, dense_layer in enumerate(self.dense_layers):
            self.add_module('nn_model_layer_{}'.format(idx_i), dense_layer)

        if self.smooth:
            self.activation_func = partial(F.elu, alpha=self.alpha_)
        else:
            self.activation_func = F.relu

        self.model_save_path = path.join(config.get('experiments', 'malware_detector') + '_' + self.name,
                                         'model.pth')

    def parse_args(self,
                   embedding_dim=64,
                   hidden_units=None,
                   dense_hidden_units=None,
                   dropout=0.6,
                   with_relu=False,
                   with_bias=True,
                   smooth=True,
                   alpha_=0.2,
                   **kwargs
                   ):
        self.embedding_dim = embedding_dim
        if hidden_units is None:
            self.hidden_units = [200, 200]
        elif isinstance(hidden_units, list):
            self.hidden_units = hidden_units
        else:
            raise TypeError("Expect a list of hidden units.")

        if dense_hidden_units is None:
            self.dense_hidden_units = [200, 200]
        elif isinstance(dense_hidden_units, list):
            self.dense_hidden_units = dense_hidden_units
        else:
            raise TypeError("Expect a list of hidden units.")

        self.dropout = dropout
        self.alpha_ = alpha_
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.smooth = smooth
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x1, binariz_x2, x2):
        # dnn
        for dense_layer in self.dense_layers[:-2]:
            x1 = self.activation_func(dense_layer(x1))
        # gcn
        binariz_x2 = binariz_x2.unsqueeze(-1) * self.embedding_weight
        binariz_x2 = self.gcn.forward(binariz_x2, x2)

        # merge
        x = torch.hstack([x1, binariz_x2])
        x = self.activation_func(self.dense_layers[-2](x))
        latent_representation = F.dropout(x, self.dropout, training=self.training)
        logits = self.dense_layers[-1](latent_representation)
        return latent_representation, logits

    def binariz_feature(self, x2):
        binary_x2 = torch.clip(torch.sum(x2, dim=-1), max=1., min=0.)
        return binary_x2, x2

    def inference(self, test_data_producer):
        confidences = []
        gt_labels = []
        self.eval()
        with torch.no_grad():
            for x1, x2, y in test_data_producer:
                x1, x2, y = utils.to_device(x1.float(), x2.float(), y.long(), self.device)
                bin_x2, x2 = self.binariz_feature(x2)
                x_hidden, logits = self.forward(x1, bin_x2, x2)
                confidences.append(F.softmax(logits, dim=-1))
                gt_labels.append(y)
            confidences = torch.vstack(confidences)
            gt_labels = torch.cat(gt_labels, dim=0)
        return confidences, gt_labels

    def get_important_attributes(self, test_data_producer, indicator_masking=False):
        """
        get important attributes by using integrated gradients
        """
        raise NotImplementedError

    def inference_batch_wise(self, x1, x2, y, use_indicator=None):
        """
        support malware samples solely
        """
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
        assert isinstance(x2, torch.Tensor)
        binariz_x2, x2 = self.binariz_feature(x2)
        _, logit = self.forward(x1, binariz_x2, x2)
        return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.size()[0],))

    def predict(self, test_data_producer, indicator_masking=False):
        """
        predict labels and conduct evaluation

        Parameters
        --------
        @param test_data_producer, torch.DataLoader
        @param indicator_masking, here is used for the purpose of compilation.
        """
        # evaluation
        confidence, y_true = self.inference(test_data_producer)
        y_pred = confidence.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        b_accuracy = balanced_accuracy_score(y_true, y_pred)
        MSG = "The accuracy on the test dataset is {:.5f}%"
        logger.info(MSG.format(accuracy * 100))
        MSG = "The balanced accuracy on the test dataset is {:.5f}%"
        logger.info(MSG.format(b_accuracy * 100))

        if np.any([np.all(y_true == i) for i in range(self.n_classes)]):
            logger.warning("class absent.")
            return

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(y_true, y_pred, average='binary')

        print("Other evaluation metrics we may need:")
        MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
        logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100))

    def customize_loss(self, logits, gt_labels, representation, mini_batch_idx):
        return F.cross_entropy(logits, gt_labels)

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=5e-4, verbose=True):
        """
        Train the malware detector, pick the best model according to the cross-entropy loss on validation set

        Parameters
        ----------
        @param train_data_producer: Object, an iterator for producing a batch of training data
        @param validation_data_producer: Object, an iterator for producing validation dataset
        @param epochs, Integer, epochs
        @param lr, Float, learning rate for Adam optimizer
        @param weight_decay, Float, penalty factor, default value 5e-4 in graph attention layer
        @param verbose: Boolean, whether to show verbose logs
        """
        optimizer = optim.Adam(self.customize_param(weight_decay), lr=lr)
        best_avg_acc = 0.
        best_epoch = 0
        total_time = 0.
        nbatches = len(train_data_producer)
        for i in range(epochs):
            self.train()
            losses, accuracies = [], []
            for idx_batch, (x1_train, x2_train, y_train) in enumerate(train_data_producer):
                x1_train, x2_train, y_train = utils.to_device(x1_train.float(), x2_train.float(), y_train.long(),
                                                              self.device)
                binariz_x2_train, x2_train = self.binariz_feature(x2_train)
                start_time = time.time()
                optimizer.zero_grad()
                latent_rpst, logits = self.forward(x1_train, binariz_x2_train, x2_train)
                loss_train = self.customize_loss(logits, y_train, latent_rpst, idx_batch)
                loss_train.backward()
                optimizer.step()
                total_time = total_time + time.time() - start_time
                acc_train = (logits.argmax(1) == y_train).sum().item()
                acc_train /= y_train.size()[0]
                mins, secs = int(total_time / 60), int(total_time % 60)
                losses.append(loss_train.item())
                accuracies.append(acc_train)
                if verbose:
                    print(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}')

            self.eval()
            avg_acc_val = []
            with torch.no_grad():
                for x1_val, x2_val, y_val in validation_data_producer:
                    x1_val, x2_val, y_val = utils.to_device(x1_val.float(), x2_val.float(), y_val.long(), self.device)
                    binariz_x2_val, x2_val = self.binariz_feature(x2_val)
                    _, logits = self.forward(x1_val, binariz_x2_val, x2_val)
                    acc_val = (logits.argmax(1) == y_val).sum().item()
                    acc_val /= x1_val.size()[0]
                    avg_acc_val.append(acc_val)
                avg_acc_val = np.mean(avg_acc_val)

            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                if not path.exists(self.model_save_path):
                    utils.mkdir(path.dirname(self.model_save_path))
                torch.save(self.state_dict(), self.model_save_path)
                if verbose:
                    print(f'Model saved at path: {self.model_save_path}')

            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def customize_param(self, weight_decay):
        customized_params_no_decay = []
        customized_params_decay = []

        for name, param in self.named_parameters():
            if 'nn_model_layer_' in name or 'embedding_weight' in name:
                customized_params_no_decay.append(param)
            else:
                customized_params_decay.append(param)
        return [{'params': customized_params_no_decay, 'weight_decay': 0.},
                {'params': customized_params_decay, 'weight_decay': weight_decay}]

    def load(self):
        """
        load model parameters from disk
        """
        self.load_state_dict(torch.load(self.model_save_path))
