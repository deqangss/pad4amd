"""
@ARTICLE{9321695,
  author={D. {Li} and Q. {Li} and Y. F. {Ye} and S. {Xu}},
  journal={IEEE Transactions on Network Science and Engineering},
  title={A Framework for Enhancing Deep Neural Networks against Adversarial Malware},
  year={2021},
  doi={10.1109/TNSE.2021.3051354}}
"""

import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, round_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.pgd')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


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
        assert norm == 'l2' or norm == 'linf', "Expect 'l2' or 'linf'."
        self.norm = norm
        self.use_random = use_random
        self.round_threshold = rounding_threshold
        self.lambda_ = 1.

    def _perturb(self, model, x, adj=None, label=None,
                 steps=10,
                 step_length=1.,
                 lambda_=1.,
                 ):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param adj: torch.FloatTensor or None, adjacency matrix (if not None, the shape is [number_of_graphs, batch_size, vocab_dim, vocab_dim])
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of iterations
        @param step_length: float, the step length in each iteration
        @param lambda_, float, penalty factor
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x
        self.lambda_ = lambda_
        self.padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        model.eval()
        for t in range(steps):
            if t == 0 and self.use_random:
                adv_x = get_x0(adv_x, rounding_threshold=self.round_threshold, is_sample=True)
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            hidden, logit = model.forward(var_adv_x, adj)
            loss, done = self.get_loss(model, logit, label, hidden, self.lambda_)
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0]
            perturbation = self.get_perturbation(grad, x, adv_x)
            adv_x = torch.clamp(adv_x + perturbation * step_length, min=0., max=1.)
        # print(torch.topk(torch.abs(adv_x - x), k=100, dim=-1))
        with torch.no_grad():
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_loss(model, logit, label, hidden, self.lambda_)
            logger.info(f"\t continuous pgd {self.norm}: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%:{self.lambda_}.")
        # round
        if self.norm == 'linf':
            # see paper: Adversarial Deep Learning for Robust Detection of Binary Encoded Malware
            round_threshold = torch.rand(adv_x.size()).to(self.device)
        else:
            round_threshold = 0.5
        # print(torch.topk(torch.abs(round_x(adv_x, round_threshold) - x), k=100, dim=-1))
        return round_x(adv_x, round_threshold)

    def perturb(self, model, x, adj=None, label=None,
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
        self.lambda_ = min_lambda_
        if 'k' in list(model.__dict__.keys()) and model.k > 0:
            logger.warning("The attack leads to dense graph and trigger the issue of out of memory.")
        adv_x = x.detach().clone().to(torch.float)
        while self.lambda_ <= max_lambda_:
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_loss(model, logit, label, hidden, self.lambda_)
            if torch.all(done):
                break
            adv_x[~done] = x[~done]  # recompute the perturbation under other penalty factors
            adv_adj = None if adj is None else adj[~done]
            pert_x = self._perturb(model, adv_x[~done], adv_adj, label[~done],
                                   steps,
                                   step_length,
                                   lambda_=self.lambda_
                                   )
            adv_x[~done] = pert_x
            self.lambda_ *= base
            if not self.check_lambda(model):
                break

        with torch.no_grad():
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_loss(model, logit, label, hidden, self.lambda_)
            if verbose:
                logger.info(f"pgd {self.norm}: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")
        return adv_x

    def get_perturbation(self, gradients, features, adv_features):
        div_zero_overflow = torch.tensor(1e-30, dtype=gradients.dtype, device=gradients.device)
        red_ind = list(range(1, len(features.size())))
        # 1. mask paddings
        gradients = gradients * self.padding_mask

        # 2. look for allowable position, because only '1--> -' and '0 --> +' are permitted
        #    2.1 api insertion
        pos_insertion = (adv_features <= 0.5) * 1 * (adv_features >= 0.)
        grad4insertion = (gradients > 0) * pos_insertion * gradients
        # grad4insertion = (gradients > 0) * gradients
        #    2.2 api removal
        pos_removal = (adv_features > 0.5) * 1
        # if self.is_attacker:
        #     #     2.2.1 cope with the interdependent apis
        #     checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
        #     grad4removal = torch.sum(gradients * checking_nonexist_api, dim=-1, keepdim=True) + gradients
        #     grad4removal *= (grad4removal < 0) * (pos_removal & self.manipulation_x)
        # else:
        grad4removal = (gradients < 0) * (pos_removal & self.manipulation_x) * gradients
        gradients = grad4removal + grad4insertion

        # 3. norm
        if self.norm == 'linf':
            perturbation = torch.sign(gradients)
        elif self.norm == 'l2':
            l2norm = torch.sqrt(torch.max(div_zero_overflow, torch.sum(gradients ** 2, dim=red_ind, keepdim=True)))
            perturbation = torch.minimum(
                torch.tensor(1., dtype=features.dtype, device=features.device),
                gradients / l2norm
            )
        else:
            raise ValueError("'l2' or 'linf' are expected.")

        # problematic
        # 5. tailor the interdependent apis, application specific
        # perturbation += torch.any(perturbation < 0, dim=-1, keepdim=True) * checking_nonexist_api * perturbation
        return perturbation

