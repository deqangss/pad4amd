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
    Projected gradient descent (ascent).

    Parameters
    ---------
    @param kappa, attack confidence
    @param manipulation_z, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, kappa=10., manipulation_z=None, omega=None, device=None):
        super(PGDAdam, self).__init__(manipulation_z, omega, device)
        self.kappa = kappa
        self.lambda_ = 1.

    def perturb(self, model, x, adj=None, label=None,
                steps=10,
                lr=1.,
                lambda_=1.,
                use_random=False,
                verbose=False):
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
        @param use_random, Boolean,  whether use random start point
        @param verbose, Boolean, whether present attack information or not
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x.detach().clone().to(torch.float)
        self.lambda_ = lambda_
        self.padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        if use_random:
            adv_x = rand_x(adv_x, is_sample=use_random)
        padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1
        adv_x.requires_grad = True
        optimizer = torch.optim.Adam([adv_x], lr=lr)
        model.eval()
        for t in range(steps):
            optimizer.zero_grad()

            hidden, logit = model.forward(adv_x, adj)
            loss, _ = self.get_losses(model, logit, label, hidden)
            loss = -1 * torch.mean(loss)
            loss.backward()
            grad = adv_x.grad.data * padding_mask
            pos_insertion = (adv_x < 0.5) * 1 * (adv_x >= 0.)
            grad4insertion = (grad < 0) * pos_insertion * grad  # owing to gradient descent
            pos_removal = (adv_x >= 0.5) * 1 * (adv_x <= 1.)
            grad4removal = (grad > 0) * (pos_removal & self.manipulation_z) * grad
            adv_x.grad.data = (grad4removal + grad4insertion).data
            adv_x.grad.data = grad.data
            optimizer.step()
            adv_x.data = adv_x.data.clamp(min=0., max=1.)

        print(torch.sum(torch.abs(adv_x.round() - x), dim=(1, 2)))
        return adv_x.round()

    def perturb_ehs(self, model, x, adj=None, label=None,
                    steps=10,
                    lr=1.,
                    min_lambda_=1e-5,
                    max_lambda_=1e5,
                    use_random=False,
                    verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        self.lambda_ = min_lambda_
        adv_x = x.detach().clone()
        while self.lambda_ <= max_lambda_:
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_losses(model, logit, label, hidden)
            if verbose:
                logger.info(
                    f"BCA attack: attack effectiveness {done.sum().item() / x.size()[0]} with lambda {self.lambda_}.")
            if torch.all(done):
                return adv_x

            adv_adj = None if adj is None else adv_adj[~done]
            pert_x = self.perturb(model, adv_x[~done], adv_adj, label[~done],
                                  steps,
                                  lr,
                                  lambda_=self.lambda_,
                                  use_random=use_random,
                                  verbose=False
                                  )
            adv_x[~done] = pert_x
            self.lambda_ *= 10.
        return adv_x

    def get_losses(self, model, logit, label, hidden=None):
        ce = F.cross_entropy(logit, label, reduction='none')
        if 'forward_g' in type(model).__dict__.keys():
            de = model.forward_g(hidden, logit.argmax(1))
            tau = model.get_tau_sample_wise(logit.argmax(1))
            loss_no_reduction = ce + \
                                self.lambda_ * (torch.clamp(
                torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW), max=self.kappa))
            # loss_no_reduction = ce + self.lambda_ * (de - model.tau)
            done = (logit.argmax(1) == 0.) & (de >= tau)
        else:
            loss_no_reduction = ce
            done = logit.argmax(1) == 0.
        return loss_no_reduction, done
