from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import numpy as np

from core.defense.maldet import MalwareDetector
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.adv_maldetector')
logger.addHandler(ErrorHandler)


class MalwareDetectorIndicator(MalwareDetector):
    def __init__(self, vocab_size, n_classes, beta=1., sigma=0.15916, sample_weights=None, n_sample_times=5, device='cpu', name='PRO', enable_gd_ckpt=False, **kwargs):
        self.beta = beta
        self.sigma = sigma
        self.sample_weights = sample_weights
        self.device = device
        self.enable_gd_ckpt = enable_gd_ckpt
        super(MalwareDetectorIndicator, self).__init__(vocab_size,
                                                       n_classes,
                                                       n_sample_times,
                                                       self.device,
                                                       name,
                                                       **kwargs)

        self.dense = nn.Linear(self.penultimate_hidden_unit + 1, self.n_classes, bias=False)
        self.phi = nn.Parameter(torch.zeros(size=(self.n_classes,)), requires_grad=False)
        if self.sample_weights is None:
            self.sample_weights = torch.ones((n_classes,), dtype=torch.float, device=self.dense)
        else:
            self.sample_weights = torch.from_numpy(np.array(self.sample_weights)).to(self.device)
        self.model_save_path = path.join(config.get('experiments', 'malware_detector_indicator') + '_' + self.name,
                                         'model.pth')
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))

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
                                       torch.ones(size=(latent_representation.shape[0], 1), dtype=torch.float32, device=self.device)])
        return latent_rep_ext, self.dense(latent_rep_ext)

    def forward_g(self, representation):
        return torch.sum(self.gaussian_prob(representation) * self.phi, dim=1)

    def update_phi(self, logits, mini_batch_idx):
        prob = torch.mean(torch.softmax(logits, dim=1), dim=0)

        if mini_batch_idx > 0:
            # a little bit non-accurate if batch size is altered during training
            self.phi += nn.Parameter(prob, requires_grad=False)
            self.phi /= 2.
        else:
            self.phi = nn.Parameter(prob, requires_grad=False)

    def gaussian_prob(self, x):
        exp_over_flow = 1e-12
        assert 0 <= self.sigma
        d = self.penultimate_hidden_unit + 1
        reverse_sigma = 1. / (self.sigma + exp_over_flow)
        det_sigma = np.power(self.sigma, d)
        prob = 1. / ((2 * np.pi) ** (d / 2.) * det_sigma ** 0.5) * torch.exp(-0.5 * torch.sum(
            (x.unsqueeze(1) - self.dense.weight) * reverse_sigma * (x.unsqueeze(1) - self.dense.weight), dim=-1))
        return prob

    def energy(self, representation, logits, sample_weights):
        exp_over_flow = 1e-12
        gamma_z = torch.softmax(logits, dim=1)
        prob_n = self.gaussian_prob(representation)

        # print(prob_n)
        # print(self.phi)
        # print(self.sample_weights)
        # print(torch.sum(prob_n * self.phi + exp_over_flow, dim=1))
        print(sample_weights)
        print(torch.sum(-torch.log(prob_n * self.phi + exp_over_flow), dim=1))
        E_z = torch.sum(torch.log(prob_n * self.phi + exp_over_flow) * sample_weights, dim=1)
        # E_z = torch.sum(gamma_z * torch.log(prob_n * self.phi / gamma_z + exp_over_flow) * sample_weights, dim=1)  # ELBO
        energies = -torch.mean(E_z, dim=0)
        return energies

    def get_sample_weights(self, labels):
        _labels = labels.numpy()
        _labels, counts = np.unique(_labels, return_counts=True)
        if _labels.shape[0] < self.n_classes:
            return torch.zeros((self.n_classes, ), dtype=torch.float, device=self.device)
        else:
            sample_weights = np.ones_like(_labels).astype(np.float32)
            _weights = float(np.max(counts)) / counts
            for i in range(_labels.shape[0]):
                sample_weights[_labels[i]] = _weights[i]
            return torch.from_numpy(sample_weights).to(self.device)

    def customize_loss(self, logits, gt_labels, representation,  mini_batch_idx):
        print(gt_labels)
        self.update_phi(logits, mini_batch_idx)
        sample_weights = self.get_sample_weights(gt_labels)
        de = self.energy(representation, logits, sample_weights) * self.beta
        ce = F.cross_entropy(logits, gt_labels)
        return de + ce



