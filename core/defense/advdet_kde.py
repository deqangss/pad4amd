from os import path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.defense.dense_est import DensityEstimator
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.advdet_kde')
logger.addHandler(ErrorHandler)


class KernelDensityEstimation(DensityEstimator):
    """
    kernel density estimation upon the penultimate layer

    parameters
    -------------
    @param model, torch.nn.Module, an instantiation of model object
    @param bandwidth, float, variance for Gaussian density function
    @param n_classes, integer, number of classes
    @param ratio, float [0,1], ratio for computing the threshold
    """

    def __init__(self, model, n_centers=1000, bandwidth=20., n_classes=2, ratio=0.9):
        super(KernelDensityEstimation, self).__init__()
        self.model = model
        self.device = model.device
        self.n_centers = n_centers
        self.bandwidth = bandwidth
        self.n_classes = n_classes
        self.ratio = ratio
        self.gaussian_means = None

        self.tau = nn.Parameter(torch.zeros([self.n_classes, ], device=self.device), requires_grad=False)
        self.name = self.model.name
        self.model.load()
        self.model_save_path = path.join(config.get('experiments', 'kde') + '_' + self.name, 'model.pth')
        self.model.model_save_path = self.model_save_path

    def forward(self, x_, adj=None):
        return self.model.forward(x_, adj)

    def forward_g(self, x_hidden, y_pred):
        """
        1 / |X_| * \sum(exp(||x - x_||_2^2/sigma^2))

        parameters
        -----------
        @param x_hidden, torch.tensor, node hidden representation
        @param y_pred, torch.tensor, prediction
        """
        size = x_hidden.size()[0]
        dist = [torch.sum(torch.square(means.unsqueeze(dim=0) - x_hidden.unsqueeze(dim=1)), dim=-1) for means in
                self.gaussian_means]
        kd = torch.stack([torch.mean(torch.exp(-d / self.bandwidth), dim=-1) for d in dist], dim=1)
        # return p(x|y=y_pred)
        return kd[torch.arange(size), y_pred]

    def get_threshold(self, validation_data_producer):
        """
        get the threshold for density estimation
        :@param validation_data_producer: Object, an iterator for producing validation dataset
        """
        self.eval()
        probabilities = []
        with torch.no_grad():
            for _ in tqdm(range(self.model.n_sample_times)):
                prob_, Y = [], []
                for x_val, adj_val, y_val, _1 in validation_data_producer:
                    x_val, adj_val, y_val = utils.to_tensor(x_val, adj_val, y_val, self.device)
                    x_hidden, logits = self.forward(x_val, adj_val)
                    x_prob = self.forward_g(x_hidden, y_val)
                    prob_.append(x_prob)
                    Y.append(y_val)
                prob_ = torch.cat(prob_)
                probabilities.append(prob_)
            prob = torch.mean(torch.stack(probabilities), dim=0)
            Y = torch.cat(Y)
            for i in range(self.n_classes):
                prob_x_y = prob[Y == i]
                s, _ = torch.sort(prob_x_y, descending=True)
                self.tau[i] = s[int((s.shape[0] - 1) * self.ratio)]

    def predict(self, test_data_producer):
        # evaluation on detector & indicator
        y_cent, x_prob, y_true = self.inference(test_data_producer)
        y_pred = y_cent.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        indicator_flag = self.indicator(x_prob, y_pred).cpu().numpy()
        # filter out examples with low likelihood
        y_pred = y_pred[indicator_flag]
        y_true = y_true[indicator_flag]
        logger.info('The indicator is turning on...')
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

    def eval(self):
        self.model.eval()

    def inference(self, test_data_producer):
        y_cent, x_prob = [], []
        gt_labels = []
        self.eval()
        with torch.no_grad():
            for ith in tqdm(range(self.model.n_sample_times)):
                y_conf_batches = []
                x_prob_batches = []
                for x, adj, y, _1 in test_data_producer:
                    x, adj, y = utils.to_tensor(x, adj, y, self.device)
                    x_hidden, logit = self.forward(x, adj)
                    y_conf_batches.append(F.softmax(logit, dim=-1))
                    x_prob_batches.append(self.forward_g(x_hidden, logit.argmax(dim=1)))
                    if ith == 0:
                        gt_labels.append(y)
                y_conf_batches = torch.vstack(y_conf_batches)
                y_cent.append(y_conf_batches)
                x_prob.append(torch.hstack(x_prob_batches))
        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.mean(torch.stack(y_cent).permute([1, 0, 2]), dim=1)
        x_prob = torch.mean(torch.stack(x_prob), dim=0)
        return y_cent, x_prob, gt_labels

    def inference_batch_wise(self, x, a, y, use_indicator=True):
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
        if a is not None:
            assert isinstance(a, torch.Tensor)
        x_hidden, logit = self.forward(x, a)
        y_pred = logit.argmax(1)
        x_prob = self.forward_g(x_hidden, y_pred)
        if use_indicator:
            return torch.softmax(logit, dim=-1).detach().cpu().numpy(), x_prob.detach().cpu().numpy()
        else:
            return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.shape[0], ))

    def get_tau_sample_wise(self, y_pred):
        return self.tau[y_pred]

    def indicator(self, x_prob, y_pred=None):
        assert y_pred is not None
        if isinstance(x_prob, np.ndarray):
            x_prob = torch.tensor(x_prob, device=self.device)
            return (x_prob >= self.get_tau_sample_wise(y_pred)).cpu().numpy()
        elif isinstance(x_prob, torch.Tensor):
            return x_prob >= self.get_tau_sample_wise(y_pred)
        else:
            raise TypeError("Tensor or numpy.ndarray are expected.")
        # res = probability.reshape(-1, 1).repeat_interleave(2, dim=1) >= self.tau
        # return res[torch.arange(res.size()[0]), y_pred]
        return

    def fit(self, train_dataset_producer, val_dataet_producer):
        X_hidden, Y = [], []
        self.eval()
        for x, a, y, _1 in train_dataset_producer:
            x, a, y = utils.to_tensor(x, a, y, self.device)
            x_hidden, _ = self.forward(x, a)
            X_hidden.append(x_hidden)
            Y.append(y)
            _, count = torch.unique(torch.cat(Y))
            if torch.min(count) >= self.n_centers:
                break
        X_hidden = torch.vstack(X_hidden)
        Y = torch.cat(Y)
        self.gaussian_means = [X_hidden[Y == i][:self.n_centers] for i in range(self.n_classes)]

        self.get_threshold(val_dataet_producer)

        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))
        self.save_to_disk()

    def load(self):
        ckpt = torch.load(self.model_save_path)
        self.gaussian_means = ckpt['gaussian_means']
        self.tau = ckpt['tau']
        self.model.load_state_dict(ckpt['base_model'])

    def save_to_disk(self):
        torch.save({
            'gaussian_means': self.gaussian_means,
            'tau': self.tau,
            'base_model': self.model.state_dict()
        },
            self.model_save_path
        )
