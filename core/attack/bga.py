"""
@inproceedings{al2018adversarial,
  title={Adversarial deep learning for robust detection of binary encoded malware},
  author={Al-Dujaili, Abdullah and Huang, Alex and Hemberg, Erik and O’Reilly, Una-May},
  booktitle={2018 IEEE Security and Privacy Workshops (SPW)},
  pages={76--82},
  year={2018},
  organization={IEEE}
}
"""

import torch
import torch.nn.functional as F
import numpy as np

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, xor_tensors, or_tensors
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.bga')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class BGA(BaseAttack):
    """
    Multi-step bit gradient ascent

    Parameters
    ---------
    @param is_attacker, Boolean, if ture means the role is the attacker
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence on adversary indicator
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(BGA, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.omega = None  # no interdependent apis if just api insertion is considered
        self.manipulation_z = None  # all apis are permitted to be insertable
        self.lambda_ = 1.

    def _perturb(self, model, x, label=None,
                 m=10,
                 lmda=1.,
                 use_sample=False):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, feature vectors with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param m: Integer, maximum number of perturbations, namely the hp k in the paper
        @param lmda, float, penalty factor for balancing the importance of adversary detector
        @param use_sample, Boolean, whether use random start point
        """
        if x is None or x.shape[0] <= 0:
            return []
        # some book-keeping
        sqrt_m = torch.from_numpy(np.sqrt([x.size()[1]])).float().to(model.device)

        adv_x = x.clone()
        worst_x = x.detach().clone()
        model.eval()
        adv_x = get_x0(adv_x, rounding_threshold=0.5, is_sample=use_sample)
        for t in range(m):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label, lmda)
            worst_x[done] = adv_x[done]
            grad = torch.autograd.grad(loss.mean(), var_adv_x)[0].data

            # filtering un-considered graphs & positions
            # grad4insertion = (grad > 0) * grad * (adv_x <= 0.5)
            # grad4ins_ = grad4insertion.reshape(x.shape[0], -1)
            # compute the updates
            x_update = (sqrt_m * (1. - 2. * adv_x) * grad >= torch.norm(
                grad, 2, 1).unsqueeze(1).expand_as(adv_x)).float()
            # find the next sample with projection to the feasible set
            adv_x = xor_tensors(x_update, adv_x)
            adv_x = or_tensors(adv_x, x)

            # select adv x
            done = self.get_scores(model, adv_x, label).data
            worst_x[done] = adv_x[done]
        return worst_x

    def perturb(self, model, x, label=None,
                steps=10,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                use_sample=False,
                base=10.,
                verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
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
                                   lmda=self.lambda_,
                                   use_sample=use_sample
                                   )
            adv_x[~done] = pert_x
            self.lambda_ *= base
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if verbose:
                logger.info(f"BGA: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3f}%.")
        return adv_x
