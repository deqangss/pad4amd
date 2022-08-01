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
from config import logging, ErrorHandler

logger = logging.getLogger('examples.base_attack')
logger.addHandler(ErrorHandler)

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
        api_flag = self.inverse_feature.get_api_flag()
        self.api_flag = torch.LongTensor(api_flag).bool().to(self.device)

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

    def produce_adv_mal(self, x_mod_list, feature_path_list, app_dir, save_dir=None):
        """
        produce adversarial malware in practice

        Parameters
        --------
        @param x_mod_list, list of tensor, each of which corresponds to the numerical modification applied to features
        @param feature_path_list, list of feature paths, each of which corresponds to the saved file of call graph
        @param app_dir, string, a directory (or a list of path) pointing to (pristine) malicious apps
        @param save_dir, directory for saving resultant apks
        """
        if len(x_mod_list) <= 0:
            return
        assert len(x_mod_list) == len(feature_path_list)
        assert isinstance(x_mod_list[0], (torch.Tensor, np.ndarray))
        if save_dir is None:
            save_dir = os.path.join('/tmp/', 'adv_mal_cache')
        if not os.path.exists(save_dir):
            utils.mkdir(save_dir)

        x_mod_instructions = [self.inverse_feature.inverse_map_manipulation(x_mod) for x_mod in x_mod_list]
        if os.path.isdir(app_dir):
            app_path_list = [os.path.join(app_dir, os.path.basename(os.path.splitext(feat_p)[0])) for \
                             feat_p in feature_path_list]
        # if not os.path.exists(os.path.join(save_dir, os.path.splitext(os.path.basename(feat_p))[0] + '_adv'))
        elif isinstance(app_dir, list):
            app_path_list = app_dir
        else:
            raise ValueError("Expect app directory or a list of paths, but got {}.".format(type(app_dir)))
        assert np.all([os.path.exists(app_path) for app_path in app_path_list]), "Unable to find all app paths."
        # for x_mod_instr, feature_path, app_path in zip(x_mod_instructions, feature_path_list, app_path_list):
        #     InverseDroidFeature.modify(x_mod_instr, feature_path, app_path, save_dir)

        pargs = [(x_mod_instr, feature_path, app_path, save_dir) for x_mod_instr, feature_path, app_path in
                 zip(x_mod_instructions, feature_path_list, app_path_list) if not os.path.exists(os.path.join(save_dir, os.path.splitext(os.path.basename(app_path))[0] + '_adv'))]
        cpu_count = multiprocessing.cpu_count() - 2 if multiprocessing.cpu_count() - 2 > 1 else 1
        pool = multiprocessing.Pool(cpu_count, initializer=utils.pool_initializer)
        for res in pool.map(InverseDroidFeature.modify_wrapper, pargs):  # keep in order
            if isinstance(res, Exception):
                logger.exception(res)
        pool.close()
        pool.join()

    def check_lambda(self, model):
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            return True
        else:
            return False

    def get_loss(self, model, adv_x, label, lambda_=None):
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(adv_x)
        else:
            logits_f = model.forward(adv_x)

        ce = F.cross_entropy(logits_f, label, reduction='none')
        y_pred = logits_f.argmax(1)
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            assert lambda_ is not None
            tau = model.get_tau_sample_wise(y_pred)
            if self.is_attacker:
                loss_no_reduction = ce + lambda_ * (torch.clamp(tau - prob_g,
                                                                max=self.kappa)
                                                    )
            else:
                loss_no_reduction = ce + lambda_ * (tau - prob_g)
            done = (y_pred != label) & (prob_g <= tau)
        else:
            loss_no_reduction = ce
            done = y_pred != label
        return loss_no_reduction, done

    def get_scores(self, model, pertb_x, label, lmda=1.):
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(pertb_x)
        else:
            logits_f = model.forward(pertb_x)
        y_pred = logits_f.argmax(1)
        if hasattr(model, 'is_detector_enabled') and (not self.oblivion):
            tau = model.get_tau_sample_wise(y_pred)
            done = (y_pred != label) & (prob_g <= tau)
        else:
            done = y_pred != label
        return done
