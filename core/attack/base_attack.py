"""
base class for waging attacks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import multiprocessing

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np

from core.droidfeature import InverseDroidFeature
from tools import utils

EXP_OVER_FLOW = 1e-30


class BaseAttack(Module):
    """
    Abstract class for attacks

    Parameters
    ---------
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, float, attack confidence
    @param manipulation_x, boolean vector shows the modifiable apis
    @param omega, list of 4 sets, each set contains the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(BaseAttack, self).__init__()
        self.is_attacker = is_attacker
        self.oblivion = oblivion
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

    def produce_adv_mal(self, x_mod_list, feature_path_list, app_dir, adj_mod=None, save_dir=None):
        """
        produce adversarial malware in practice

        Parameters
        --------
        @param x_mod_list, list of tensor, each of which corresponds to the numerical modification applied to apis in call graph
        @param feature_path_list, list of feature paths, each of which corresponds to the saved file of call graph
        @param app_dir, string, a directory (or a list of path) pointing to (pristine) malicious apps
        @param adj_mod, modifications on adjacency matrix
        @param save_dir, directory for saving resultant apks
        """
        if adj_mod is not None:
            raise NotImplementedError("Un-support to apply the modifications of adjacency matrix to apps.")
        if len(x_mod_list) <= 0:
            return
        assert len(x_mod_list) == len(feature_path_list)
        assert isinstance(x_mod_list[0], (torch.Tensor, np.ndarray))
        print('ok1')
        x_mod_instructions = [self.inverse_feature.inverse_map_manipulation(x_mod) for x_mod in x_mod_list]
        if os.path.isdir(app_dir):
            app_path_list = [os.path.join(app_dir, os.path.basename(os.path.splitext(feat_p)[0])) for \
                             feat_p in feature_path_list]
        elif isinstance(app_dir, list):
            app_path_list = app_dir
        else:
            raise ValueError("Expect app directory or a list of paths, but got {}.".format(type(app_dir)))
        assert np.all([os.path.exists(app_path) for app_path in app_path_list]), "Unable to find all app paths."

        # for x_mod_instr, feature_path, app_path in zip(x_mod_instructions, feature_path_list, app_path_list):
        #     InverseDroidFeature.modify(x_mod_instr, feature_path, app_path, save_dir)

        pargs = [(x_mod_instr, feature_path, app_path, save_dir) for x_mod_instr, feature_path, app_path in
                 zip(x_mod_instructions, feature_path_list, app_path_list)]
        cpu_count = multiprocessing.cpu_count() - 2 if multiprocessing.cpu_count() - 2 > 1 else 1
        pool = multiprocessing.Pool(cpu_count, initializer=utils.pool_initializer)
        for _ in pool.map(InverseDroidFeature.modify_wrapper, pargs):  # keep in order
            pass
        pool.close()
        pool.join()

    def check_lambda(self, model):
        if 'forward_g' in type(model).__dict__.keys() and (not self.oblivion):
            return True
        else:
            return False

    def get_loss(self, model, logit, label, hidden=None, lambda_=None):
        ce = F.cross_entropy(logit, label, reduction='none')
        y_pred = logit.argmax(1)
        if 'forward_g' in type(model).__dict__.keys() and (not self.oblivion):
            assert lambda_ is not None
            de = model.forward_g(hidden, y_pred)
            tau = model.get_tau_sample_wise(y_pred)
            if self.is_attacker:
                loss_no_reduction = ce + lambda_ * (torch.clamp(
                        torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW), max=self.kappa))
            else:
                loss_no_reduction = ce + lambda_ * (torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW))
                # loss_no_reduction = ce + self.lambda_ * (de - model.tau)
            done = (y_pred == 0.) & (de >= tau)
        else:
            loss_no_reduction = ce
            done = y_pred == 0.
        return loss_no_reduction, done
