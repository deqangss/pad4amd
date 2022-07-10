"""
@ARTICLE{9321695,
  author={D. {Li} and Q. {Li} and Y. F. {Ye} and S. {Xu}},
  journal={IEEE Transactions on Network Science and Engineering},
  title={A Framework for Enhancing Deep Neural Networks against Adversarial Malware},
  year={2021},
  doi={10.1109/TNSE.2021.3051354}}
"""

import torch

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, round_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.pgd')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120


class PGD(BaseAttack):
    """
    Projected gradient descent (ascent).

    Parameters
    ---------
    @param norm, 'l2' or 'linf'
    @param use_random, Boolean,  whether use random start point
    @param rounding_threshold, float, a threshold for rounding real scalars
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, norm, use_random=False, rounding_threshold=0.5,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(PGD, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        assert norm == 'l1' or norm == 'l2' or norm == 'linf', "Expect 'l1', 'l2' or 'linf'."
        self.norm = norm
        self.use_random = use_random
        assert 0 < rounding_threshold < 1
        self.round_threshold = rounding_threshold
        self.lambda_ = 1.

    def _perturb(self, model, x, label=None,
                 steps=10,
                 step_length=1.,
                 lambda_=1.,
                 ):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of iterations
        @param step_length: float, the step length in each iteration
        @param lambda_, float, penalty factor
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x
        self.lambda_ = lambda_
        model.eval()
        loss_natural = 0.
        for t in range(steps):
            if t == 0 and self.use_random:
                adv_x = get_x0(adv_x, rounding_threshold=self.round_threshold, is_sample=True)
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label, self.lambda_)
            if t == 0:
                loss_natural = loss
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0].detach().data
            perturbation = self.get_perturbation(grad, x, adv_x)
            adv_x = torch.clamp(adv_x + perturbation * step_length, min=0., max=1.)
        # round
        if self.norm == 'linf' and (not hasattr(model, 'is_detector_enabled')):
            round_threshold = torch.rand(x.size()).to(self.device)
        else:
            round_threshold = self.round_threshold
        adv_x = round_x(adv_x, round_threshold)
        loss_adv, _1 = self.get_loss(model, adv_x, label, self.lambda_)
        replace_flag = (loss_adv < loss_natural).unsqueeze(1).expand_as(adv_x)
        adv_x[replace_flag] = x[replace_flag]
        return adv_x

    def perturb(self, model, x, label=None,
                steps=10,
                step_length=1.,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                base=10.,
                verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        if 'k' in list(model.__dict__.keys()) and model.k > 0:
            logger.warning("The attack leads to dense graph and trigger the issue of out of memory.")
        assert steps >= 0 and step_length >= 0
        model.eval()
        if hasattr(model, 'is_detector_enabled'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        adv_x = x.detach().clone().to(torch.double)
        while self.lambda_ <= max_lambda_:
            with torch.no_grad():
                _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if torch.all(done):
                break
            pert_x = self._perturb(model, adv_x[~done], label[~done],
                                   steps,
                                   step_length,
                                   lambda_=self.lambda_
                                   )
            adv_x[~done] = pert_x
            self.lambda_ *= base
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if verbose:
                logger.info(f"pgd {self.norm}: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")
        return adv_x

    def get_perturbation(self, gradients, features, adv_features):
        # look for allowable position, because only '1--> -' and '0 --> +' are permitted
        # api insertion
        pos_insertion = (adv_features <= 0.5) * 1 * (adv_features >= 0.)
        grad4insertion = (gradients >= 0) * pos_insertion * gradients
        # grad4insertion = (gradients > 0) * gradients
        # api removal
        pos_removal = (adv_features > 0.5) * 1
        grad4removal = (gradients < 0) * (pos_removal & self.manipulation_x) * gradients
        if self.is_attacker:
            # cope with the interdependent apis
            checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
            grad4removal[:, self.api_flag] += torch.sum(gradients * checking_nonexist_api, dim=-1, keepdim=True)
        gradients = grad4removal + grad4insertion

        # norm
        if self.norm == 'linf':
            perturbation = torch.sign(gradients)
        elif self.norm == 'l2':
            l2norm = torch.linalg.norm(gradients, dim=-1, keepdim=True)
            perturbation = torch.minimum(
                torch.tensor(1., dtype=features.dtype, device=features.device),
                gradients / l2norm
            )
            perturbation = torch.where(torch.isnan(perturbation), 0., perturbation)
            perturbation = torch.where(torch.isinf(perturbation), 1., perturbation)
        else:
            raise ValueError("Expect 'l2' or 'linf' norm.")

        # add the extra perturbation owing to the interdependent apis
        if self.norm == 'linf' and self.is_attacker:
            perturbation += torch.any(perturbation[:, self.api_flag] < 0, dim=-1,
                                      keepdim=True) * checking_nonexist_api
        if self.norm == 'l2' and self.is_attacker:
            min_val = torch.amin(perturbation, dim=-1, keepdim=True).clamp_(max=0.)
            perturbation += (torch.any(perturbation[:, self.api_flag] < 0, dim=-1,
                                       keepdim=True) * torch.abs(min_val) * checking_nonexist_api)
        return perturbation
