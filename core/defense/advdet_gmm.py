from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from captum.attr import IntegratedGradients

import numpy as np

from core.defense.maldet import MalwareDetector
from core.defense.dense_est import DensityEstimator
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.advdet_gmm')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class MalwareDetectorIndicator(MalwareDetector, DensityEstimator):
    def __init__(self, vocab_size, n_classes, beta=1., sigma=0.1416, ratio=0.90, sample_weights=None, n_sample_times=5,
                 device='cpu', name='', enable_gd_ckpt=False, **kwargs):
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
        DensityEstimator.__init__(self)

        self.dense = nn.Linear(self.penultimate_hidden_unit + 1, self.n_classes, bias=False)
        self.phi = nn.Parameter(torch.zeros(size=(self.n_classes,)), requires_grad=False)
        self.tau = nn.Parameter(torch.zeros(size=[], dtype=torch.float), requires_grad=False)
        if self.sample_weights is None:
            self.sample_weights = torch.ones((n_classes,), dtype=torch.float, device=self.dense)
        else:
            self.sample_weights = torch.from_numpy(np.array(self.sample_weights)).to(self.device)
        self.model_save_path = path.join(config.get('experiments', 'gmm') + '_' + self.name,
                                         'model.pth')

    def predict(self, test_data_producer, indicator_masking=False):
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
        if not indicator_masking:
            # filter out examples with low likelihood
            # y_pred = y_pred[indicator_flag]
            # y_true = y_true[indicator_flag]
            flag_of_retaining = indicator_flag | (y_pred == 1.)  # excluding the examples with ``not sure'' response
            y_pred = y_pred[flag_of_retaining]
            y_true = y_true[flag_of_retaining]
        else:
            # instead filtering out examples, here resets the prediction as 1
            y_pred[~indicator_flag] = 1.
        logger.info('The indicator is turning on...')
        measurement(y_true, y_pred)

    def inference(self, test_data_producer):
        y_cent, x_prob = [], []
        gt_labels = []
        self.eval()
        with torch.no_grad():
            for ith in tqdm(range(self.n_sample_times)):
                y_cent_batches = []
                x_prob_batches = []
                for x, adj, y, _1 in test_data_producer:
                    x, adj, y = utils.to_tensor(x, adj, y, self.device)
                    x_hidden, logits = self.forward(x, adj)
                    y_cent_batches.append(F.softmax(logits, dim=-1))
                    x_prob_batches.append(self.forward_g(x_hidden))
                    if ith == 0:
                        gt_labels.append(y)
                y_cent_batches = torch.vstack(y_cent_batches)
                y_cent.append(y_cent_batches)
                x_prob.append(torch.hstack(x_prob_batches))

        gt_labels = torch.cat(gt_labels, dim=0)
        y_cent = torch.mean(torch.stack(y_cent).permute([1, 0, 2]), dim=1)
        x_prob = torch.mean(torch.stack(x_prob), dim=0)
        return y_cent, x_prob, gt_labels

    def get_important_attributes(self, test_data_producer, indicator_masking=False):
        """
        get important attributes by using integrated gradients

        adjacency matrix is neglected
        """
        attributions_cls = []
        attributions_de = []

        def _ig_wrapper_cls(_x):
            _3, logits = self.forward(_x, adj=None)
            return F.softmax(logits, dim=-1)

        ig_cls = IntegratedGradients(_ig_wrapper_cls)

        def _ig_wrapper_de(_x):
            x_hidden, _4 = self.forward(_x, adj=None)
            return self.forward_g(x_hidden)

        ig_de = IntegratedGradients(_ig_wrapper_de)

        for i, (x, _1, y, _2) in enumerate(test_data_producer):
            x, _4, y = utils.to_tensor(x, None, y, self.device)
            x.requires_grad = True
            base_lines = torch.zeros_like(x, dtype=torch.float32, device=self.device)
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

    def inference_batch_wise(self, x, a, y, use_indicator=True):
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
        if a is not None:
            assert isinstance(a, torch.Tensor)
        self.eval()
        x_hidden, logit = self.forward(x, a)
        x_prob = self.forward_g(x_hidden)
        if use_indicator:
            return torch.softmax(logit, dim=-1).detach().cpu().numpy(), x_prob.detach().cpu().numpy()
        else:
            return torch.softmax(logit, dim=-1).detach().cpu().numpy(), np.ones((logit.shape[0],))

    def get_tau_sample_wise(self, y_pred=None):
        return self.tau

    def indicator(self, x_prob, y_pred=None):
        if isinstance(x_prob, np.ndarray):
            x_prob = torch.tensor(x_prob, device=self.device)
            return (x_prob >= self.tau).cpu().numpy()
        elif isinstance(x_prob, torch.Tensor):
            return x_prob >= self.tau
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
                for x_val, adj_val, y_val, _ in validation_data_producer:
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

    def forward(self, x_feature, adj=None):
        if self.enable_gd_ckpt:
            x_feature.requires_grad = True
            if adj is not None:
                adj.requires_grad = True
            latent_representation = checkpoint(self.malgat, x_feature, adj)  # saving RAM dramatically
        else:
            latent_representation = self.malgat(x_feature, adj)
        latent_representation = F.dropout(latent_representation, self.dropout, training=self.training)
        hidden = torch.hstack([latent_representation, torch.ones(size=(latent_representation.shape[0], 1),
                                                                 dtype=torch.float32,
                                                                 device=self.device)])
        return hidden, self.dense(hidden)

    def forward_g(self, x_hidden, y_pred=None):
        return torch.sum(self.gaussian_prob(x_hidden) * self.phi, dim=1)

    def update_phi(self, logits, mini_batch_idx):
        cent = torch.mean(torch.softmax(logits, dim=1), dim=0)

        if mini_batch_idx > 0:
            # a little bit non-accurate if batch size is altered during training
            self.phi += nn.Parameter(cent, requires_grad=False)
            self.phi /= 2.
        else:
            self.phi = nn.Parameter(cent, requires_grad=False)

    def gaussian_prob(self, x):
        assert 0 <= self.sigma
        d = self.penultimate_hidden_unit + 1
        reverse_sigma = 1. / (self.sigma + EXP_OVER_FLOW)
        det_sigma = np.power(self.sigma, d)
        prob = 1. / ((2 * np.pi) ** (d / 2.) * det_sigma ** 0.5) * torch.exp(-0.5 * torch.sum(
            (x.unsqueeze(1) - self.dense.weight) * reverse_sigma * (x.unsqueeze(1) - self.dense.weight), dim=-1))
        return prob

    def energy(self, hidden, logits):
        gamma_z = torch.softmax(logits, dim=1)
        prob_n = self.gaussian_prob(hidden)

        E_z = torch.sum(gamma_z * torch.log(prob_n * self.phi / (gamma_z + EXP_OVER_FLOW) + \
                                            EXP_OVER_FLOW) * self.sample_weights, dim=1)  # ELBO
        return torch.mean(-E_z, dim=0)

    def customize_loss(self, logits, gt_labels, hidden, mini_batch_idx):
        self.update_phi(logits, mini_batch_idx)

        de = self.energy(hidden, logits) * self.beta
        ce = F.cross_entropy(logits, gt_labels)
        return de + ce

    def load(self):
        # load model
        self.load_state_dict(torch.load(self.model_save_path))

    def save_to_disk(self):
        assert path.exists(self.model_save_path), 'train model first'
        torch.save(self.state_dict(), self.model_save_path)
