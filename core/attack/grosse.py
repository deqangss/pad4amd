"""
@inproceedings{grosse2017adversarial,
  title={Adversarial examples for malware detection},
  author={Grosse, Kathrin and Papernot, Nicolas and Manoharan, Praveen and Backes, Michael and McDaniel, Patrick},
  booktitle={European symposium on research in computer security},
  pages={62--79},
  year={2017},
  organization={Springer}
}
"""

import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.grosse')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class Groose(BaseAttack):
    """
    Multi-step bit coordinate ascent applied upon softmax output (rather than upon the loss)

    Parameters
    ---------
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(Groose, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.omega = None  # no interdependent apis if just api insertion is considered
        self.manipulation_z = None  # all apis are insertable
        self.lambda_ = 1.

    def _perturb(self, model, x, adj=None, label=None,
                 m=10,
                 lambda_=1.):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param adj: torch.FloatTensor or None, adjacency matrix (if not None, the shape is [number_of_graphs, batch_size, vocab_dim, vocab_dim])
        @param label: torch.LongTensor, ground truth labels
        @param m: Integer, maximum number of perturbations
        @param lambda_, float, penalty factor
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x
        self.lambda_ = lambda_
        padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        model.eval()
        for t in range(m):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            hidden, logit = model.forward(var_adv_x, adj)
            loss, done = self.get_loss(model, logit, label, hidden)
            if torch.all(done):
                break
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0].data

            # filtering un-considered graphs & positions
            grad = grad * padding_mask
            grad4insertion = (grad > 0) * grad * (adv_x <= 0.5)

            grad4ins_ = grad4insertion.reshape(x.shape[0], -1)
            _, pos = torch.max(grad4ins_, dim=-1)
            perturbation = F.one_hot(pos, num_classes=grad4ins_.shape[-1]).float().reshape(x.shape)
            # avoid to perturb the examples that are successful to evade the victim
            perturbation[done] = 0.
            adv_x = torch.clamp(adv_x + perturbation, min=0., max=1.)
        return adv_x

    def perturb(self, model, x, adj=None, label=None,
                m=10,
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
            _, done = self.get_loss(model, logit, label, hidden)
            if torch.all(done):
                break
            adv_x[~done] = x[~done]  # recompute the perturbation under other penalty factors
            adv_adj = None if adj is None else adj[~done]
            pert_x = self._perturb(model, adv_x[~done], adv_adj, label[~done],
                                   m,
                                   lambda_=self.lambda_
                                   )
            adv_x[~done] = pert_x
            self.lambda_ *= base
            if not self.check_lambda(model):
                break
        with torch.no_grad():
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_loss(model, logit, label, hidden)
            if verbose:
                logger.info(f"grosse: attack effectiveness {done.sum().item() / x.size()[0]}.")

        return adv_x

    def get_loss(self, model, logit, label, hidden=None):
        softmax_loss = torch.softmax(logit, dim=-1)[torch.arange(label.size()[0]), 0]
        y_pred = logit.argmax(1)
        if 'forward_g' in type(model).__dict__.keys() and (not self.oblivion):
            de = model.forward_g(hidden, y_pred)
            tau = model.get_tau_sample_wise(y_pred)
            if self.is_attacker:
                loss_no_reduction = softmax_loss + self.lambda_ * (torch.clamp(
                        torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW), max=self.kappa))
            else:
                loss_no_reduction = softmax_loss + self.lambda_ * (torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW))

            done = (y_pred == 0.) & (de >= tau)
        else:
            loss_no_reduction = softmax_loss
            done = y_pred == 0.
        return loss_no_reduction, done
