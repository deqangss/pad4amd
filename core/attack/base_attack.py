"""
base class for waging attacks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

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
    @param manipulation_x, boolean vector shows the modifiable apis
    @param omega, list of 4 sets, each set contains the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, kappa=1., manipulation_x=None, omega=None, device=None):
        super(BaseAttack, self).__init__()
        self.kappa = kappa
        self.manipulation_x = manipulation_x
        self.device = device
        self.omega = omega
        self.inverse_feature = InverseDroidFeature()
        self.padding_mask = None
        self.initialize()

    def initialize(self):
        if self.manipulation_x is None:
            self.manipulation_x = self.inverse_feature.get_manipulation()
        self.manipulation_x = torch.LongTensor(self.manipulation_x).to(self.device)
        if self.omega is None:
            self.omega = self.inverse_feature.get_interdependent_apis()
        print(type(self.omega))
        print(type(self.inverse_feature.vocab))
        self.omega = torch.sum(
            F.one_hot(torch.tensor(self.omega), num_classes=len(self.inverse_feature.vocab)),
            dim=0).to(self.device)

    def perturb(self, model, x, adj=None, label=None):
        """
        perturb node feature vectors

        Parameters
        --------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors, each represents an api
        @param adj: torch.FloatTensor or None, adjacency matrix (if not None, the shape is [number_of_graphs, batch_size, vocab_dim, vocab_dim])
        @param label: torch.LongTensor, ground truth labels
        """
        raise NotImplementedError

    def produce_adv_mal(self, x_mod_list, feature_path_list, app_dir, adj_mod=None):
        """
        produce adversarial malware in practice

        Parameters
        --------
        @param x_mod_list, list of tensor, each of which corresponds to the numerical modification applied to apis in call graph
        @param feature_path_list, list of feature paths, each of which corresponds to the saved file of call graph
        @param app_dir, string, a directory (or a list of path) pointing to (pristine) malicious apps
        @param adj_mod, modifications on adjacency matrix
        """
        if adj_mod is not None:
            raise NotImplementedError("Un-support to apply the modifications of adjacency matrix to apps.")
        if len(x_mod_list) <= 0:
            return
        assert len(x_mod_list) == len(feature_path_list)
        assert isinstance(x_mod_list[0], (torch.Tensor, np.ndarray))

        x_mod_instructions = [self.inverse_feature.inverse_map_manipulation(x_mod) for x_mod in x_mod_list]
        if os.path.isdir(app_dir):
            app_path_list = [os.path.join(app_dir, os.path.basename(os.path.splitext(feat_p)[0])) for \
                             feat_p in feature_path_list]
        elif isinstance(app_dir, list):
            app_path_list = app_dir
        else:
            raise ValueError("Expect app dir or paths, but got {}.".format(app_dir))
        assert np.all([os.path.exists(app_path) for app_path in app_path_list])

        for x_mod_instr, feature_path, app_path in zip(x_mod_instructions, feature_path_list, app_path_list):
            self.inverse_feature.modify(x_mod_instr, feature_path, app_path)

    @staticmethod
    def check_lambda(model):
        if 'forward_g' in type(model).__dict__.keys():
            return True
        else:
            return False

    def get_loss(self, model, logit, label, hidden=None, lambda_=None):
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
