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
from tools.utils import get_x0
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
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, use_random=False, rounding_threshold=0.98,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(PGDAdam, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.use_random = use_random
        self.round_threshold = rounding_threshold
        self.lambda_ = 1.

    def _perturb(self, model, x, adj=None, label=None,
                 steps=10,
                 lr=1.,
                 lambda_=1.,
                 adam_state=None):
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
            adv_x = get_x0(adv_x, rounding_threshold=self.round_threshold, is_sample=True)
        padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1
        adv_x.requires_grad = True
        optimizer = torch.optim.Adam([adv_x], lr=lr)
        print('adv:', adv_x.shape)


        if adam_state is not None:
            optimizer.load_state_dict(adam_state)
        model.eval()
        for t in range(steps):
            optimizer.zero_grad()
            hidden, logit = model.forward(adv_x, adj)
            loss, _ = self.get_loss(model, logit, label, hidden, self.lambda_)
            loss = -1 * torch.mean(loss)  # optimizer is a type of gradient descent method
            loss.backward()
            grad = adv_x.grad * padding_mask
            pos_insertion = (adv_x <= 0.5) * 1 * (adv_x >= 0.)
            grad4insertion = (
                                     grad < 0) * pos_insertion * grad  # positions of gradient value smaller than zero are used for insertion
            pos_removal = (adv_x > 0.5) * 1 * (adv_x <= 1.)
            grad4removal = (grad > 0) * (pos_removal & self.manipulation_x) * grad
            adv_x.grad = (grad4removal + grad4insertion)
            adv_x.grad = grad
            optimizer.step()
            adv_x.data = adv_x.data.clamp(min=0., max=1.)
        print(adam_state['state'][0]['exp_avg'].shape)
        return adv_x.detach(), optimizer.state_dict()

    def perturb(self, model, x, adj=None, label=None,
                steps=10,
                lr=1.,
                step_check=10,
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
        self.lambda_ = min_lambda_
        assert steps >= 0 and step_check > 0 and lr >= 0
        mini_steps = [step_check] * (steps // step_check)
        mini_steps = mini_steps + [steps % step_check] if steps % step_check != 0 else mini_steps

        adv_x = x.detach().clone().to(torch.float)
        while self.lambda_ <= max_lambda_:
            pert_x_cont = None
            prev_done = None
            adam_state = None
            for i, mini_step in enumerate(mini_steps):
                hidden, logit = model.forward(adv_x, adj)
                _, done = self.get_loss(model, logit, label, hidden, self.lambda_)
                if torch.all(done):
                    break
                if i == 0:
                    adv_x[~done] = x[~done]  # recompute the perturbation under other penalty factors
                    adv_adj = None if adj is None else adj[~done]
                    prev_done = done
                else:
                    adv_x[~done] = pert_x_cont[~done[~prev_done]]
                    adv_adj = None if adj is None else adj[~done]
                    prev_done = done
                    # adam_state['state'][0]['exp_avg'] = adam_state['state'][0]['exp_avg'][~done[~prev_done]]
                    # adam_state['state'][0]['exp_avg_sq'] = adam_state['state'][0]['exp_avg_sq'][~done[~prev_done]]
                    # print(adam_state['state'][0]['exp_avg'].shape)
                    # print('adv:', adv_x[~done].shape)
                print(i)
                pert_x_cont, adam_state = self._perturb(model, adv_x[~done], adv_adj, label[~done],
                                                        mini_step,
                                                        lr,
                                                        lambda_=self.lambda_,
                                                        adam_state=None
                                                        )
                # round
                adv_x[~done] = pert_x_cont.round()

            self.lambda_ *= base
            if not self.check_lambda(model):
                break
        with torch.no_grad():
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_loss(model, logit, label, hidden, self.lambda_)
            if verbose:
                logger.info(f"pgd adam attack: attack effectiveness {done.sum().item() / x.size()[0] * 100}%.")
        return adv_x
