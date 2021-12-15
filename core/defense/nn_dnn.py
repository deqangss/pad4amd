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

from core.defense.malgat import MalGAT
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.dnn')
logger.addHandler(ErrorHandler)


class DNNMalwareDetector(nn.Module):
    def __init__(self, input_size, n_classes, device='cpu', name='DNN', **kwargs):
        """
        Construct malware detector

        Parameters
        ----------
        @param input_size: Integer, the dimentionality number of input vector
        @param n_classes: Integer, the number of classes, n=2
        @param device: String, 'cpu' or 'cuda'
        @param name: String, model name
        """
        super(DNNMalwareDetector, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.device = device
        self.name = name

        self.parse_args(**kwargs)

        self.dense_layers = []
        self.dense_layers.append(nn.Linear(self.input_size, self.dense_hidden_units[0]))
        for i in range(len(self.dense_hidden_units[0:-1])):
            self.dense_layers.append(nn.Linear(self.dense_hidden_units[i],  # start from idx=1
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
                   dense_hidden_units=None,
                   dropout=0.6,
                   alpha_=0.2,
                   smooth=False,
                   **kwargs
                   ):
        if dense_hidden_units is None:
            self.dense_hidden_units = [200, 200]
        elif isinstance(dense_hidden_units, list):
            self.dense_hidden_units = dense_hidden_units
        else:
            raise TypeError("Expect a list of hidden units.")

        self.dropout = dropout
        self.alpha_ = alpha_
        self.smooth = smooth
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x):
        """
        Go through the neural network

        Parameters
        ----------
        @param x1: 2D tensor, feature representation
        """
        for dense_layer in self.dense_layers[:-1]:
            x = self.activation_func(dense_layer(x))

        latent_representation = F.dropout(x, self.dropout, training=self.training)
        logits = self.dense_layers[-1](latent_representation)
        return latent_representation, logits

    def binariz_feature(self, x1, x2):
        binary_x2 = torch.clip(torch.sum(x2, dim=-1), max=1., min=0.)
        return torch.hstack([x1, binary_x2])

    def inference(self, test_data_producer):
        confidences = []
        gt_labels = []
        self.eval()
        with torch.no_grad():
            for x1, x2, y in test_data_producer:
                x1, x2, y = utils.to_device(x1.float(), x2.float(), y.long(), self.device)
                x = self.binariz_feature(x1, x2)
                x_hidden, logits = self.forward(x)
                confidences.append(F.softmax(logits, dim=-1))
                gt_labels.append(y)
        confidences = torch.vstack(confidences)
        gt_labels = torch.cat(gt_labels, dim=0)
        return confidences, gt_labels

    def get_important_attributes(self, test_data_producer, target_label=1):
        """
        get important attributes by using integrated gradients
        """
        attributions = []
        gt_labels = []

        def _ig_wrapper(_x):
            _3, logits = self.forward(_x)
            return F.softmax(logits, dim=-1)
        ig = IntegratedGradients(_ig_wrapper)

        for i, (x1, x2, y) in enumerate(test_data_producer):
            x1, x2, y = utils.to_tensor(x1, x2, y)
            x = self.binariz_feature(x1, x2)
            x.requires_grad = True
            attribution_bs = ig.attribute(x,
                                          baselines=torch.zeros_like(x, dtype=torch.float32, device=self.device),
                                          target=target_label)
            attributions.append(attribution_bs.clone().detach().cpu().numpy())
            gt_labels.append(y.clone().detach().cpu().numpy())
            np.save('./labels', np.concatenate(gt_labels))
        return np.vstack(attributions)

    def inference_batch_wise(self, x1, x2, y):
        assert isinstance(x1, torch.Tensor) and isinstance(y, torch.Tensor)
        assert isinstance(x2, torch.Tensor)
        x = self.binariz_feature(x1, x2)
        _, logit = self.forward(x)
        return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.size()[0],))

    def predict(self, test_data_producer):
        """
        predict labels and conduct evaluation

        Parameters
        --------
        @param test_data_producer, torch.DataLoader
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

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., verbose=True):
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
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_avg_acc = 0.
        best_epoch = 0
        total_time = 0.
        nbatches = len(train_data_producer)
        for i in range(epochs):
            self.train()
            losses, accuracies = [], []
            for idx_batch, (x1_train, x2_train, y_train) in enumerate(train_data_producer):
                x1_train, x2_train, y_train = utils.to_device(x1_train.float(), x2_train.float(), y_train.long(), self.device)
                x_train = self.binariz_feature(x1_train, x2_train)
                start_time = time.time()
                optimizer.zero_grad()
                latent_rpst, logits = self.forward(x_train)
                loss_train = self.customize_loss(logits, y_train, latent_rpst, idx_batch)
                loss_train.backward()
                optimizer.step()
                total_time = total_time + time.time() - start_time
                acc_train = (logits.argmax(1) == y_train).sum().item()
                acc_train /= x_train.size()[0]
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
                    x_val = self.binariz_feature(x1_val, x2_val)
                    _, logits = self.forward(x_val)
                    acc_val = (logits.argmax(1) == y_val).sum().item()
                    acc_val /= x_val.size()[0]
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

    def load(self):
        """
        load model parameters from disk
        """
        self.load_state_dict(torch.load(self.model_save_path))
