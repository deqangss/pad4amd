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
from tools.utils import rand_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.grosse')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = -30


class Groose(BaseAttack):
    """
    Multi-step bit coordinate ascent using softmax output (rather than the loss)

    Parameters
    ---------
    @param kappa, attack confidence
    @param manipulation_z, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, kappa=10., manipulation_z=None, omega=None, device=None):
        super(Groose, self).__init__(manipulation_z, omega, device)
        self.omega = None  # no interdependent apis if just api insertion is considered
        self.manipulation_z = None  # all apis are insertable
        self.kappa = kappa
        self.lambda_ = 1.

    def perturb(self, model, x, adj=None, label=None,
                m_perturbations=10,
                lambda_=1.,
                use_sample=False,
                stop=True,
                verbose=False):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param adj: torch.FloatTensor or None, adjacency matrix (if not None, the shape is [number_of_graphs, batch_size, vocab_dim, vocab_dim])
        @param label: torch.LongTensor, ground truth labels
        @param m_perturbations: Integer, maximum number of perturbations
        @param lambda_, float, penalty factor
        @param use_sample, Boolean, whether use random start point
        @param stop, Boolean, whether stop once evade victim successfully
        @param verbose, Boolean, whether present attack information or not
        """
        if x is None and x.shape[0] <= 0:
            return []
        adv_x = x.detach().clone().to(torch.float)
        self.lambda_ = lambda_
        padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        model.eval()
        for t in range(m_perturbations):
            if use_sample and t == 0:
                adv_x = rand_x(adv_x, is_sample=True)
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            hidden, logit = model.forward(var_adv_x, adj)
            loss, done = self.get_losses(model, logit, hidden)
            if verbose:
                print(
                    f"\n Iteration {t}: the accuracy is {(logit.argmax(1) == 1.).sum().item() / adv_x.size()[0] * 100:.3f}.")
            if torch.all(done) and stop:
                break
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0].data

            # filtering un-considered graphs & positions
            grad = grad * padding_mask
            grad4insertion = (grad > 0) * grad * (adv_x < 0.5)

            grad4ins_ = grad4insertion.reshape(x.shape[0], -1)
            _, pos = torch.max(grad4ins_, dim=-1)
            perturbation = F.one_hot(pos, num_classes=grad4ins_.shape[-1]).float().reshape(x.shape)
            if stop:
                perturbation[done] = 0.
            adv_x = torch.clamp(adv_x + perturbation, min=0., max=1.)
        return adv_x

    def perturb_ehs(self, model, x, adj=None, label=None,
                    m_perturbations=10,
                    min_lambda_=1e-5,
                    max_lambda_=1e5,
                    use_sample=False,
                    stop=True,
                    verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        self.lambda_ = min_lambda_
        adv_node = x.detach().clone()
        while self.lambda_ <= max_lambda_:
            hidden, logit = model.forward(adv_node, adj)
            _, done = self.get_losses(model, logit, hidden)
            if verbose:
                logger.info(
                    f"BCA attack: attack effectiveness {done.sum().item() / x.size()[0]} with lambda {self.lambda_}.")
            if torch.all(done):
                return adv_node

            adv_adj = None if adj is None else adv_adj[~done]
            pert_x = self.perturb(model, adv_node[~done], adv_adj, label[~done],
                                  m_perturbations,
                                  lambda_=self.lambda_,
                                  use_sample=use_sample,
                                  stop=stop,
                                  verbose=False
                                  )
            adv_node[~done] = pert_x
            self.lambda_ *= 10.
        return adv_node

    def get_losses(self, model, logit, hidden=None):
        softmax_loss = torch.softmax(logit, dim=-1)[:, 0]
        if 'forward_g' in type(model).__dict__.keys():
            de = model.forward_g(hidden, logit.argmax(1))
            tau = model.get_tau_sample_wise(logit.argmax(1))
            loss_no_reduction = softmax_loss + \
                                self.lambda_ * (torch.clamp(
                torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW), max=self.kappa))
            done = (logit.argmax(1) == 0.) & (de >= tau)
        else:
            loss_no_reduction = softmax_loss
            done = logit.argmax(1) == 0.
        return loss_no_reduction, done
