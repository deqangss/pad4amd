from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import warnings
import os.path as path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from captum.attr import IntegratedGradients

import numpy as np

from core.defense.md_dnn import MalwareDetectionDNN
from core.defense.amd_template import DetectorTemplate
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.amd_input_convex_nn')
logger.addHandler(ErrorHandler)


class AdvMalwareDetectorICNN(nn.Module, DetectorTemplate):
    def __init__(self, md_nn_model, input_size, n_classes, ratio=0.95,
                 device='cpu', name='', **kwargs):
        nn.Module.__init__(self)
        DetectorTemplate.__init__(self)
        self.input_size = input_size
        self.n_classes = n_classes
        self.ratio = ratio
        self.device = device
        self.name = name
        self.parse_args(**kwargs)
        if isinstance(md_nn_model, nn.Module):
            self.md_nn_model = md_nn_model
        else:
            kwargs['smooth'] = True
            self.md_nn_model = MalwareDetectionDNN(self.input_size,
                                                   n_classes,
                                                   self.device,
                                                   name,
                                                   **kwargs)
            warnings.warn("Use a self-defined NN-based malware detector")
        if hasattr(self.md_nn_model, 'smooth'):
            if not self.md_nn_model.smooth:  # non-smooth, exchange it
                for name, child in self.md_nn_model.named_children():
                    if isinstance(child, nn.ReLU):
                        self.md_nn_model._modules['relu'] = nn.SELU()
        else:
            for name, child in self.md_nn_model.named_children():
                if isinstance(child, nn.ReLU):
                    self.md_nn_model._modules['relu'] = nn.SELU()
        self.md_nn_model = self.md_nn_model.to(self.device)

        # input convex neural network
        self.non_neg_dense_layers = []
        if len(self.dense_hidden_units) < 1:
            raise ValueError("Expect at least one hidden layer.")
        for i in range(len(self.dense_hidden_units[0:-1])):
            self.non_neg_dense_layers.append(nn.Linear(self.dense_hidden_units[i],  # start from idx=1
                                                       self.dense_hidden_units[i + 1],
                                                       bias=False))
        self.non_neg_dense_layers.append(nn.Linear(self.dense_hidden_units[-1], 1, bias=False))
        # registration
        for idx_i, dense_layer in enumerate(self.non_neg_dense_layers):
            self.add_module('non_neg_layer_{}'.format(idx_i), dense_layer)

        self.dense_layers = []
        self.dense_layers.append(nn.Linear(self.input_size, self.dense_hidden_units[0]))
        for i in range(len(self.dense_hidden_units[1:])):
            self.dense_layers.append(nn.Linear(self.input_size, self.dense_hidden_units[i]))
        self.dense_layers.append(nn.Linear(self.input_size, 1))
        # registration
        for idx_i, dense_layer in enumerate(self.dense_layers):
            self.add_module('layer_{}'.format(idx_i), dense_layer)

        self.tau = nn.Parameter(torch.zeros([1, ], device=self.device), requires_grad=False)

        self.model_save_path = path.join(config.get('experiments', 'amd_icnn') + '_' + self.name,
                                         'model.pth')
        logger.info('========================================icnn model architecture==============================')
        logger.info(self)
        logger.info('===============================================end==========================================')

    def parse_args(self,
                   dense_hidden_units=None,
                   dropout=0.6,
                   alpha_=0.2,
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
        self.proc_number = kwargs['proc_number']
        if len(kwargs) > 0:
            logger.warning("Unknown hyper-parameters {}".format(str(kwargs)))

    def forward_f(self, x):
        return self.md_nn_model(x)

    def forward_g(self, x):
        prev_x = None
        for i, dense_layer in enumerate(self.dense_layers):
            x_add = []
            x1 = dense_layer(x)
            x_add.append(x1)
            if prev_x is not None:
                x2 = self.non_neg_dense_layers[i - 1](prev_x)
                x_add.append(x2)
            prev_x = torch.sum(torch.stack(x_add, dim=0), dim=0)
            if i < len(self.dense_layers):
                prev_x = F.selu(prev_x)
        return prev_x.reshape(-1)

    def forward(self, x):
        return self.forward_f(x), self.forward_g(x)

    def predict(self, test_data_producer, indicator_masking=True):
        """
        predict labels and conduct evaluation on detector & indicator

        Parameters
        --------
        @param test_data_producer, torch.DataLoader
        @param indicator_masking, whether filtering out the examples with low density or masking their values
        """
        y_cent, x_prob, y_true = self.inference(test_data_producer)
        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        indicator_flag = self.indicator(x_prob).cpu().numpy()

        def measurement(_y_true, _y_pred):
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
            accuracy = accuracy_score(_y_true, _y_pred)
            b_accuracy = balanced_accuracy_score(_y_true, _y_pred)
            MSG = "The accuracy on the test dataset is {:.5f}%"
            logger.info(MSG.format(accuracy * 100))
            MSG = "The balanced accuracy on the test dataset is {:.5f}%"
            logger.info(MSG.format(b_accuracy * 100))

            if np.any([np.all(_y_true == i) for i in range(self.n_classes)]):
                logger.warning("class absent.")
                return

            tn, fp, fn, tp = confusion_matrix(_y_true, _y_pred).ravel()
            fpr = fp / float(tn + fp)
            fnr = fn / float(tp + fn)
            f1 = f1_score(_y_true, _y_pred, average='binary')
            print("Other evaluation metrics we may need:")
            MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
            logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100))

        measurement(y_true, y_pred)

        rtn_value = (y_pred == 0) & indicator_flag

        if indicator_masking:
            # excluding the examples with ``not sure'' response
            y_pred = y_pred[indicator_flag]
            y_true = y_true[indicator_flag]
        else:
            # instead filtering out examples, here resets the prediction as 1
            y_pred[~indicator_flag] = 1.
        logger.info('The indicator is turning on...')
        logger.info('The threshold is {:.5}'.format(self.tau.item()))
        measurement(y_true, y_pred)

        return rtn_value

    def inference(self, test_data_producer):
        y_cent, x_prob = [], []
        gt_labels = []
        self.eval()
        with torch.no_grad():
            for x, y in test_data_producer:
                x, y = utils.to_device(x.double(), y.long(), self.device)
                logits_f, logits_g = self.forward(x)
                y_cent.append(torch.softmax(logits_f, dim=-1))
                x_prob.append(logits_g)
                gt_labels.append(y)

        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.cat(y_cent, dim=0)
        x_prob = torch.cat(x_prob, dim=0)
        return y_cent, x_prob, gt_labels

    def get_important_attributes(self, test_data_producer, indicator_masking=False):
        """
        get important attributes by using integrated gradients

        adjacency matrix is neglected
        """
        attributions_cls = []
        attributions_de = []

        def _ig_wrapper_cls(_x):
            logits = self.forward_f(_x)
            return F.softmax(logits, dim=-1)

        ig_cls = IntegratedGradients(_ig_wrapper_cls)

        def _ig_wrapper_de(_x):
            return self.forward_g(_x)

        ig_de = IntegratedGradients(_ig_wrapper_de)

        for i, (x, y) in enumerate(test_data_producer):
            x, y = utils.to_tensor(x, y, self.device)
            x.requires_grad = True
            base_lines = torch.zeros_like(x, dtype=torch.double, device=self.device)
            base_lines[:, -1] = 1
            attribution_bs = ig_cls.attribute(x,
                                              baselines=base_lines,
                                              target=1)
            attributions_cls.append(attribution_bs.clone().detach().cpu().numpy())

            attribution_bs = ig_de.attribute(x,
                                             baselines=base_lines
                                             )
            attributions_de.append(attribution_bs.clone().detach().cpu().numpy())
        return np.vstack(attributions_cls), np.vstack(attributions_de)

    def inference_batch_wise(self, x):
        assert isinstance(x, torch.Tensor)
        self.eval()
        logits_f, logits_g = self.forward(x)
        return torch.softmax(logits_f, dim=-1).detach().cpu().numpy(), logits_g.detach().cpu().numpy()

    def get_tau_sample_wise(self, y_pred=None):
        return self.tau

    def indicator(self, x_prob, y_pred=None):
        """
        Return 'True' if a sample is original, and otherwise 'False' is returned.
        """
        if isinstance(x_prob, np.ndarray):
            x_prob = torch.tensor(x_prob, device=self.device)
            return (x_prob <= self.tau).cpu().numpy()
        elif isinstance(x_prob, torch.Tensor):
            return x_prob <= self.tau
        else:
            raise TypeError("Tensor or numpy.ndarray are expected.")

    def get_threshold(self, validation_data_producer, ratio=None):
        """
        get the threshold for adversary detection
        :@param validation_data_producer: Object, an iterator for producing validation dataset
        """
        self.eval()
        ratio = ratio if ratio is not None else self.ratio
        assert 0 <= ratio <= 1
        probabilities = []
        with torch.no_grad():
            for x_val, y_val in validation_data_producer:
                x_val, y_val = utils.to_tensor(x_val.double(), y_val.long(), self.device)
                x_logits = self.forward_g(x_val)
                probabilities.append(x_logits)
            s, _ = torch.sort(torch.cat(probabilities, dim=0))
            i = int((s.shape[0] - 1) * ratio)
            assert i >= 0
            self.tau[0] = s[i]

    def reset_threshold(self):
        self.tau[0] = 0.

    def customize_loss(self, logits_x, labels, logits_adv_x, labels_adv, beta_1=1, beta_2=1):
        if logits_adv_x is not None and len(logits_adv_x) > 0:
            G = F.binary_cross_entropy_with_logits(logits_adv_x, labels_adv)
        else:
            G = 0
        if logits_x is not None and len(logits_x) > 0:
            F_ = F.cross_entropy(logits_x, labels)
        else:
            F_ = 0
        return beta_1 * F_ + beta_2 * G

    def fit(self, train_data_producer, validation_data_producer, epochs=100, lr=0.005, weight_decay=0., verbose=True):
        """
        Train the malware & adversary detector, pick the best model according to the validation results

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
                # make data for training g
                # 1. add pepper and salt noises
                x_train_noises = torch.clamp(x_train + utils.psn(x_train, np.random.uniform(0, 0.5)),
                                             min=0., max=1.)
                x_train_ = torch.cat([x_train, x_train_noises], dim=0)
                y_train_ = torch.cat([torch.zeros(x_train.shape[:1]), torch.ones(x_train.shape[:1])]).double().to(
                    self.device)
                idx = torch.randperm(y_train_.shape[0])
                x_train_ = x_train_[idx]
                y_train_ = y_train_[idx]

                start_time = time.time()
                optimizer.zero_grad()
                logits_f = self.forward_f(x_train)
                logits_g = self.forward_g(x_train_)
                loss_train = self.customize_loss(logits_f, y_train, logits_g, y_train_)
                loss_train.backward()
                optimizer.step()
                # clamp
                constraint = utils.NonnegWeightConstraint()
                for name, module in self.named_modules():
                    if 'non_neg_layer' in name:
                        module.apply(constraint)
                total_time = total_time + time.time() - start_time
                acc_f_train = (logits_f.argmax(1) == y_train).sum().item()
                acc_f_train /= x_train.size()[0]
                acc_g_train = ((F.sigmoid(logits_g) >= 0.5) == y_train_).sum().item()
                acc_g_train /= x_train_.size()[0]

                mins, secs = int(total_time / 60), int(total_time % 60)
                losses.append(loss_train.item())
                accuracies.append(acc_f_train)
                accuracies.append(acc_g_train)
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + idx_batch + 1}/{epochs * nbatches} | training time in {mins:.0f} minutes, {secs} seconds.')
                    logger.info(
                        f'Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {acc_f_train * 100:.2f}% & {acc_g_train * 100:.2f}%.')

            self.eval()
            avg_acc_val = []
            with torch.no_grad():
                for x_val, y_val in validation_data_producer:
                    x_val, y_val = utils.to_device(x_val.double(), y_val.long(), self.device)
                    x_val_noises = torch.clamp(x_val + utils.psn(x_val, np.random.uniform(0, 0.5)),
                                               min=0., max=1.)
                    x_val_ = torch.cat([x_val, x_val_noises], dim=0)
                    y_val_ = torch.cat([torch.zeros(x_val.shape[:1]), torch.ones(x_val.shape[:1])]).long().to(
                        self.device)
                    logits_f = self.forward_f(x_val)
                    logits_g = self.forward_g(x_val_)
                    acc_val = (logits_f.argmax(1) == y_val).sum().item()
                    acc_val /= x_val.size()[0]
                    avg_acc_val.append(acc_val)

                    acc_val_g = ((F.sigmoid(logits_g) >= 0.5) == y_val_).sum().item()
                    acc_val_g /= x_val_.size()[0]
                    avg_acc_val.append(acc_val_g)
                avg_acc_val = np.mean(avg_acc_val)

            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                self.get_threshold(validation_data_producer)

                self.save_to_disk()
                if verbose:
                    print(f'Model saved at path: {self.model_save_path}')

            if verbose:
                logger.info(
                    f'Training loss (epoch level): {np.mean(losses):.4f} | Train accuracy: {np.mean(accuracies) * 100:.2f}')
                logger.info(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def load(self):
        # load model
        assert path.exists(self.model_save_path), 'train model first'
        # ckpt = torch.load(self.model_save_path)
        # self.tau = ckpt['tau']
        # self.md_nn_model.load_state_dict(ckpt['md_model'])
        self.load_state_dict(torch.load(self.model_save_path))

    def save_to_disk(self):
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        # torch.save({
        #     'tau': self.tau,
        #     'md_model': self.md_nn_model.state_dict(),
        #     'amd_model': self.state_dict()
        # }, self.model_save_path
        # )
        torch.save(self.state_dict(), self.model_save_path)

