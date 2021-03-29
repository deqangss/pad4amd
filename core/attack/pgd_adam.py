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
from tools.utils import rand_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.pgd')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class PGDAdam(BaseAttack):
    """
    optimize the perturbation using adam optimizer

    Parameters
    ---------
    @param use_random, Boolean,  whether use random start point
    @param rounding_threshold, float, a threshold for rounding real scalars
    @param kappa, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, use_random=False, rounding_threshold=0.98, kappa=1., manipulation_x=None, omega=None, device=None):
        super(PGDAdam, self).__init__(kappa, manipulation_x, omega, device)
        self.use_random = use_random
        self.round_threshold = rounding_threshold
        self.lambda_ = 1.

    def _perturb(self, model, x, adj=None, label=None,
                 steps=10,
                 lr=1.,
                 lambda_=1.):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param adj: torch.FloatTensor or None, adjacency matrix (if not None, the shape is [number_of_graphs, batch_size, vocab_dim, vocab_dim])
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of iterations
        @param lr: float, learning rate
        @param lambda_, float, penalty factor
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x.detach()
        self.lambda_ = lambda_
        self.padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        if self.use_random:
            adv_x = rand_x(adv_x, rounding_threshold=self.round_threshold, is_sample=True)
        padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1
        adv_x.requires_grad = True
        optimizer = torch.optim.Adam([adv_x], lr=lr)
        model.eval()
        for t in range(steps):
            optimizer.zero_grad()
            hidden, logit = model.forward(adv_x, adj)
            loss, _ = self.get_loss(model, logit, label, hidden, self.lambda_)
            loss = -1 * torch.mean(loss)  # optimizer is a type of gradient descent method
            loss.backward()
            grad = adv_x.grad * padding_mask
            pos_insertion = (adv_x < 0.5) * 1 * (adv_x >= 0.)
            grad4insertion = (grad < 0) * pos_insertion * grad  # positions of gradient value smaller than zero are used for insertion
            pos_removal = (adv_x >= 0.5) * 1 * (adv_x <= 1.)
            grad4removal = (grad > 0) * (pos_removal & self.manipulation_x) * grad
            adv_x.grad = (grad4removal + grad4insertion)
            adv_x.grad = grad
            optimizer.step()
            adv_x.data = adv_x.data.clamp(min=0., max=1.)

        print(torch.sum(torch.abs(adv_x.round() - x), dim=(1, 2)))
        return adv_x.round().detach()

    def perturb(self, model, x, adj=None, label=None,
                steps=10,
                lr=1.,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                base=10.,
                verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        self.lambda_ = min_lambda_
        adv_x = x.detach().clone().to(torch.float)
        while self.lambda_ <= max_lambda_:
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_loss(model, logit, label, hidden, self.lambda_)
            if verbose:
                logger.info(f"PGD adam attack: attack effectiveness {done.sum().item() / float(x.size()[0]):.3f} with lambda {self.lambda_}.")
            if torch.all(done):
                break
            adv_x[~done] = x[~done]  # recompute the perturbation under other penalty factors
            adv_adj = None if adj is None else adv_adj[~done]
            pert_x = self._perturb(model, adv_x[~done], adv_adj, label[~done],
                                   steps,
                                   lr,
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
                logger.info(
                    f"pgd adam attack: attack effectiveness {done.sum().item() / x.size()[0]}.")
        return adv_x
