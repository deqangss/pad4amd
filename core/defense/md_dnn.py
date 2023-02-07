from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os.path as path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import numpy as np

from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.dnn')
logger.addHandler(ErrorHandler)


class MalwareDetectionDNN(nn.Module):
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
        super(MalwareDetectionDNN, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.device = device
        self.name = name

        self.parse_args(**kwargs)

        self.dense_layers = []
        if len(self.dense_hidden_units) >= 1:
            self.dense_layers.append(nn.Linear(self.input_size, self.dense_hidden_units[0]))
        else:
            raise ValueError("Expect at least one hidden layer.")
        for i in range(len(self.dense_hidden_units[0:-1])):
            self.dense_layers.append(nn.Linear(self.dense_hidden_units[i],  # start from idx=1
                                               self.dense_hidden_units[i + 1]))
        self.dense_layers.append(nn.Linear(self.dense_hidden_units[-1], self.n_classes))
        # registration
        for idx_i, dense_layer in enumerate(self.dense_layers):
            self.add_module('nn_model_layer_{}'.format(idx_i), dense_layer)

        if self.smooth:
            self.activation_func = F.selu  # partial(F.elu, alpha=self.alpha_)
        else:
            self.activation_func = F.relu

        self.model_save_path = path.join(config.get('experiments', 'md_dnn') + '_' + self.name,
                                         'model.pth')
        logger.info('========================================dnn model architecture===============================')
        logger.info(self)
        logger.info('===============================================end==========================================')

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
        self.proc_number = kwargs['proc_number']
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward(self, x):
        """
        Go through the neural network

        Parameters
        ----------
        @param x: 2D tensor, feature representation
        """
        for dense_layer in self.dense_layers[:-1]:
            x = self.activation_func(dense_layer(x))

        latent_representation = F.dropout(x, self.dropout, training=self.training)
        logits = self.dense_layers[-1](latent_representation)
        return logits

    def inference(self, test_data_producer):
        confidences = []
        gt_labels = []
        self.eval()
        with torch.no_grad():
            for x, y in test_data_producer:
                x, y = utils.to_device(x.double(), y.long(), self.device)
                logits = self.forward(x)
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
            logits = self.forward(_x)
            return F.softmax(logits, dim=-1)
        ig = IntegratedGradients(_ig_wrapper)

        for i, (x, y) in enumerate(test_data_producer):
            x, y = utils.to_device(x.double(), y.long(), self.device)
            x.requires_grad = True
            baseline = torch.zeros_like(x, dtype=torch.double, device=self.device)
            attribution_bs = ig.attribute(x,
                                          baselines=baseline,
                                          target=target_label)
            attribution = torch.hstack(attribution_bs)
            attributions.append(attribution.clone().detach().cpu().numpy())
            gt_labels.append(y.clone().detach().cpu().numpy())
            np.save('./labels', np.concatenate(gt_labels))
        return np.vstack(attributions)

    def inference_batch_wise(self, x):
        """
        support malware samples solely
        """
        assert isinstance(x, torch.Tensor)
        logit = self.forward(x)
        return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.size()[0],))

    def predict(self, test_data_producer, indicator_masking=True):
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

    def customize_loss(self, logits, gt_labels, representation = None, mini_batch_idx=None):
        return F.cross_entropy(logits, gt_labels)

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., weight_sampling=0.5, verbose=True):
        """
        Train the malware detector, pick the best model according to the cross-entropy loss on validation set

        Parameters
        ----------
        @param train_data_producer: Object, an iterator for producing a batch of training data
        @param validation_data_producer: Object, an iterator for producing validation dataset
        @param epochs, Integer, epochs
        @param lr, Float, learning rate for Adam optimizer
        @param weight_decay, Float, penalty factor
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
            for idx_batch, (x_train, y_train) in enumerate(train_data_producer):
                x_train, y_train = utils.to_device(x_train.double(), y_train.long(), self.device)
                start_time = time.time()
                optimizer.zero_grad()
                logits = self.forward(x_train)
                loss_train = self.customize_loss(logits, y_train)
                loss_train.backward()
                optimizer.step()
                total_time = total_time + time.time() - start_time
                acc_train = (logits.argmax(1) == y_train).sum().item()
                acc_train /= x_train.size()[0]
                mins, secs = int(total_time / 60), int(total_time % 60)
                losses.append(loss_train.item())
                accuracies.append(acc_train)
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_train * 100:.2f}')

            self.eval()
            avg_acc_val = []
            with torch.no_grad():
                for x_val, y_val in validation_data_producer:
                    x_val, y_val = utils.to_device(x_val.double(), y_val.long(), self.device)
                    logits = self.forward(x_val)
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
