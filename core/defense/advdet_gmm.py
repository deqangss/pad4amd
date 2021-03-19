from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import numpy as np

from core.defense.maldet import MalwareDetector
from core.defense.dense_est import DenseEstimator
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.advdet_gmm')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class MalwareDetectorIndicator(MalwareDetector, DenseEstimator):
    def __init__(self, vocab_size, n_classes, beta=1., sigma=0.1416, ratio=0.99, sample_weights=None, n_sample_times=5,
                 device='cpu', name='PRO', enable_gd_ckpt=False, **kwargs):
        self.beta = beta
        self.sigma = sigma
        self.ratio = ratio
        self.sample_weights = sample_weights
        self.device = device
        self.enable_gd_ckpt = enable_gd_ckpt
        MalwareDetector.__init__(self, vocab_size,
                                 n_classes,
                                 n_sample_times,
                                 self.device,
                                 name,
                                 **kwargs)
        DenseEstimator.__init__(self)

        self.dense = nn.Linear(self.penultimate_hidden_unit + 1, self.n_classes, bias=False)
        self.phi = nn.Parameter(torch.zeros(size=(self.n_classes,)), requires_grad=False)
        self.tau = nn.Parameter(torch.zeros(size=[], dtype=torch.float), requires_grad=False)
        if self.sample_weights is None:
            self.sample_weights = torch.ones((n_classes,), dtype=torch.float, device=self.dense)
        else:
            self.sample_weights = torch.from_numpy(np.array(self.sample_weights)).to(self.device)
        self.model_save_path = path.join(config.get('experiments', 'malware_detector_indicator') + '_' + self.name,
                                         'model.pth')

    def predict(self, test_data_producer, use_indicator=True):
        # evaluation on detector & indicator
        confidence, probability, y_true = self.inference(test_data_producer)
        y_pred = confidence.argmax(1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        indicator_flag = self.indicator(probability).cpu().numpy()
        # filter out examples with low likelihood
        if use_indicator:
            # indicator_flag = x_prob >= self.tau.cpu().numpy()
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

    def inference(self, test_data_producer):
        confidences = []
        x_probabilities = []
        gt_labels = []
        self.eval()
        with torch.no_grad():
            for ith in tqdm(range(self.n_sample_times)):
                conf_batches = []
                x_prob_batches = []
                for res in test_data_producer:
                    x, adj, y = res
                    x, adj, y = utils.to_tensor(x, adj, y, self.device)
                    x_hidden, logits = self.forward(x, adj)
                    conf_batches.append(F.softmax(logits, dim=-1))
                    x_prob_batches.append(self.forward_g(x_hidden))
                    if ith == 0:
                        gt_labels.append(y)
                conf_batches = torch.vstack(conf_batches)
                confidences.append(conf_batches)
                x_probabilities.append(torch.hstack(x_prob_batches))
        gt_labels = torch.cat(gt_labels, dim=0)
        confidences = torch.mean(torch.stack(confidences).permute([1, 0, 2]), dim=1)
        probabilities = torch.mean(torch.stack(x_probabilities), dim=0)
        return confidences, probabilities, gt_labels

    def inference_batch_wise(self, x, a, y, use_indicator=True):
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
        if a is not None:
            assert isinstance(a, torch.Tensor)
        x_hidden, logit = self.forward(x, a)
        x_dense = self.forward_g(x_hidden)
        if use_indicator:
            return torch.softmax(logit, dim=-1).detach().cpu().numpy(), x_dense.detach().cpu().numpy()
        else:
            return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.shape[0], ))

    def get_tau_sample_wise(self, y_pred=None):
        return self.tau

    def indicator(self, x_dense, y_pred=None):
        if isinstance(x_dense, np.ndarray):
            x_dense = torch.tensor(x_dense, device=self.device)
            return (x_dense >= self.tau).cpu().numpy()
        elif isinstance(x_dense, torch.Tensor):
            return x_dense >= self.tau
        else:
            raise TypeError("Tensor or numpy.ndarray are expected.")

    def get_threshold(self, validation_data_producer):
        """
        get the threshold for density estimation
        :@param validation_data_producer: Object, an iterator for producing validation dataset
        """
        self.eval()
        probabilities = []
        with torch.no_grad():
            for _ in tqdm(range(self.n_sample_times)):
                prob_ = []
                for res in validation_data_producer:
                    x_val, adj_val, y_val = res
                    x_val, adj_val, y_val = utils.to_tensor(x_val, adj_val, y_val, self.device)
                    x_hidden, logits = self.forward(x_val, adj_val)
                    x_prob = self.forward_g(x_hidden)
                    prob_.append(x_prob)
                prob_ = torch.cat(prob_)
                probabilities.append(prob_)
            s, _ = torch.sort(torch.mean(torch.stack(probabilities), dim=0), descending=True)
            i = int((s.shape[0] - 1) * self.ratio)
            assert i >= 0
            self.tau = nn.Parameter(s[i], requires_grad=False)

    def forward(self, feature, adj=None):
        if self.enable_gd_ckpt:
            feature.requires_grad = True
            if adj is not None:
                adj.requires_grad = True
            latent_representation = checkpoint(self.malgat, feature, adj)  # saving RAM dramatically
        else:
            latent_representation = self.malgat(feature, adj)
        latent_representation = F.dropout(latent_representation, self.dropout, training=self.training)
        latent_rep_ext = torch.hstack([latent_representation,
                                       torch.ones(size=(latent_representation.shape[0], 1), dtype=torch.float32,
                                                  device=self.device)])
        return latent_rep_ext, self.dense(latent_rep_ext)

    def forward_g(self, x_hidden, y_pred=None):
        return torch.sum(self.gaussian_prob(x_hidden) * self.phi, dim=1)

    def update_phi(self, logits, mini_batch_idx):
        prob = torch.mean(torch.softmax(logits, dim=1), dim=0)

        if mini_batch_idx > 0:
            # a little bit non-accurate if batch size is altered during training
            self.phi += nn.Parameter(prob, requires_grad=False)
            self.phi /= 2.
        else:
            self.phi = nn.Parameter(prob, requires_grad=False)

    def gaussian_prob(self, x):
        assert 0 <= self.sigma
        d = self.penultimate_hidden_unit + 1
        reverse_sigma = 1. / (self.sigma + EXP_OVER_FLOW)
        det_sigma = np.power(self.sigma, d)
        prob = 1. / ((2 * np.pi) ** (d / 2.) * det_sigma ** 0.5) * torch.exp(-0.5 * torch.sum(
            (x.unsqueeze(1) - self.dense.weight) * reverse_sigma * (x.unsqueeze(1) - self.dense.weight), dim=-1))
        return prob

    def energy(self, representation, logits):
        gamma_z = torch.softmax(logits, dim=1)
        prob_n = self.gaussian_prob(representation)

        # print(prob_n)
        # print(self.sample_weights)
        debug = torch.sum(prob_n * self.phi + EXP_OVER_FLOW, dim=1)
        print(debug)
        # print(torch.sum(-torch.log(prob_n * self.phi + exp_over_flow), dim=1))
        # E_z = torch.sum(torch.log(prob_n * self.phi + exp_over_flow) * self.sample_weights, dim=1)
        E_z = torch.sum(gamma_z * torch.log(prob_n * self.phi / (gamma_z + EXP_OVER_FLOW) + \
                                            EXP_OVER_FLOW) * self.sample_weights, dim=1)  # ELBO
        return torch.mean(-E_z, dim=0)

    def get_sample_weights(self, labels):
        _labels = labels.cpu().numpy()
        _labels, counts = np.unique(_labels, return_counts=True)
        if _labels.shape[0] < self.n_classes:
            return torch.zeros((self.n_classes,), dtype=torch.float, device=self.device)
        else:
            sample_weights = np.ones_like(_labels).astype(np.float32)
            _weights = float(np.max(counts)) / counts
            for i in range(_labels.shape[0]):
                sample_weights[_labels[i]] = _weights[i]
            return torch.from_numpy(sample_weights).to(self.device)

    def customize_loss(self, logits, gt_labels, representation, mini_batch_idx):
        # print(gt_labels)
        self.update_phi(logits, mini_batch_idx)

        de = self.energy(representation, logits) * self.beta
        ce = F.cross_entropy(logits, gt_labels)
        return de + ce

    def load(self):
        # load model
        self.load_state_dict(torch.load(self.model_save_path))

    def save_to_disk(self):
        assert path.exists(self.model_save_path), 'train model first'
        torch.save(self.state_dict(), self.model_save_path)
