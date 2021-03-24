"""
base class for waging attacks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np

from core.droidfeature import InverseDroidFeature

EXP_OVER_FLOW = 1e-30


class BaseAttack(Module):
    """
    Abstract class for attacks

    Parameters
    ---------
    @param kappa, float, attack confidence
    @param manipulation_z, boolean vector shows the modifiable apis
    @param omega, list of 4 sets, each set contains the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, kappa=10, manipulation_z=None, omega=None, device=None):
        super(BaseAttack, self).__init__()
        self.kappa = kappa
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

    def perturb(self, model, x, adj=None, label=None):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors, each represents an api
        @param adj: torch.FloatTensor or None, adjacency matrix (if not None, the shape is [number_of_graphs, batch_size, vocab_dim, vocab_dim])
        @param label: torch.LongTensor, ground truth labels
        """
        raise NotImplementedError

    def realistic_adv_mal(self, x_mod, feature_path_list, app_path_list, adj_mod=None):
        """
        produce adversarial malware in realistic
        """
        assert isinstance(x_mod, (torch.Tensor, np.ndarray)) & isinstance(x_mod, (torch.Tensor, np.ndarray))
        if isinstance(x_mod, torch.Tensor):
            x_mod = x_mod.detach().cpu().numpy()


    @staticmethod
    def check_lambda(model):
        if 'forward_g' in type(model).__dict__.keys():
            return True
        else:
            return False

    def get_losses(self, model, logit, label, hidden=None, lambda_=None):
        ce = F.cross_entropy(logit, label, reduction='none')
        y_pred = logit.argmax(1)
        if 'forward_g' in type(model).__dict__.keys():
            assert lambda_ is not None
            de = model.forward_g(hidden, y_pred)
            tau = model.get_tau_sample_wise(y_pred)
            loss_no_reduction = ce + lambda_ * (torch.clamp(
                    torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW), max=self.kappa))
            # loss_no_reduction = ce + self.lambda_ * (de - model.tau)
            done = (y_pred == 0.) & (de >= tau)
        else:
            loss_no_reduction = ce
            done = y_pred == 0.
        return loss_no_reduction, done
