"""
base class for waging attacks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

from core.droidfeature import InverseDroidFeature


class BaseAttack(Module):
    """
    Abstract class for attacks

    Parameters
    ---------
    @param n_perturbations: Integer, number of perturbations
    @param manipulation_z, boolean vector shows the modifiable apis
    @param omega, list of 4 sets, each set contains the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """
    def __init__(self, n_perturbations=10, manipulation_z=None, omega=None, device=None):
        super(BaseAttack, self).__init__()
        self.n_perturbations = n_perturbations
        self.manipulation_z = manipulation_z
        self.device = device
        self.omega = omega
        self.inverse_feature = InverseDroidFeature()
        self.padding_mask = None
        self.initialize()

    def initialize(self):
        """
        todo: initialize necessaries
        """
        if self.manipulation_z is None:
            self.manipulation_z = self.inverse_feature.get_manipulation()
        self.manipulation_z = torch.LongTensor(self.manipulation_z).to(self.device)
        if self.omega is None:
            self.omega = self.inverse_feature.get_interdependent_apis()
        self.omega = torch.sum(
            F.one_hot(torch.tensor(self.omega), num_classes=len(self.inverse_feature.vocab)),
            dim=0).to(self.device)

    def perturb(self, model, node_features, label=None):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param node_features: torch.FloatTensor, node feature vectors, each represents an api
        @param label: torch.LongTensor, ground truth labels
        """
        raise NotImplementedError

    def realistic_adv_mal(self):
        """
        todo: produce adversarial malware in realistic
        """
        pass