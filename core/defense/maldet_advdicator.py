from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from tqdm import tqdm
import os.path as path

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from core.defense.maldet import MalwareDetector
from config import config, logging, ErrorHandler
from tools import utils

logger = logging.getLogger('core.defense.adv_maldetector')
logger.addHandler(ErrorHandler)


class MalwareDetectorIndicator(MalwareDetector):
    def __init__(self, vocab_size, n_classes, beta=1., sigma=0.7071, n_sample_times=5, device='cpu', name='PRO', **kwargs):
        self.beta = beta
        self.sigma = sigma
        self.device = device
        super(MalwareDetectorIndicator, self).__init__(vocab_size,
                                                       n_classes,
                                                       n_sample_times,
                                                       self.device,
                                                       name,
                                                       **kwargs)

        self.dense = nn.Linear(self.penultimate_hidden_unit + 1, self.n_classes, bias=False)
        self.phi = nn.Parameter(torch.zeros(size=(self.n_classes,)), requires_grad=False)
        self.model_save_path = path.join(config.get('experiments', 'malware_detector_indicator') + '_' + self.name,
                                         'model.pth')
        if not path.exists(self.model_save_path):
            utils.mkdir(path.dirname(self.model_save_path))

    def forward(self, feature, adj=None):
        latent_representation = self.malgat(feature, adj)
        latent_representation = F.dropout(latent_representation, self.dropout, training=self.training)
        latent_rep_ext = torch.hstack([latent_representation,
                                       torch.ones(size=(latent_representation.shape[0], 1), dtype=torch.float32, device=self.device)])
        return latent_rep_ext, self.dense(latent_rep_ext)

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

    def energy(self, representation, logits):
        exp_over_flow = 1e-12
        prob_n = self.gaussian_prob(representation)
        gamma_z = torch.softmax(logits, dim=1)
        print(prob_n)
        print(self.phi)
        print(torch.sum(prob_n * self.phi / gamma_z, dim=1))
        E_z = torch.log(torch.sum(prob_n * self.phi / gamma_z, dim=1) + exp_over_flow)
        energies = -torch.mean(E_z, dim=0)
        return energies

    def customize_loss(self, logits, gt_labels, representation,  mini_batch_idx):
        self.update_phi(logits, mini_batch_idx)
        de = self.energy(representation, logits) * self.beta
        print(de)
        ce = F.cross_entropy(logits, gt_labels)
        return de + ce



