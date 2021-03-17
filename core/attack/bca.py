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

from core.attack.base_attack import BaseAttack
from tools.utils import rand_x

EXP_OVER_FLOW = -30


class BCA(BaseAttack):
    """
    Multi-step bit coordinate ascent

    Parameters
    ---------
    @param kappa, attack confidence
    @param manipulation_z, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, kappa=10., manipulation_z=None, omega=None, device=None):
        super(BCA, self).__init__(manipulation_z, omega, device)
        self.omega = None  # no interdependent apis if just api insertion is considered
        self.manipulation_z = None  # all apis are insertable
        self.kappa = kappa
        self.lambda_ = 1.

    def perturb(self, model, x, adj=None, label=None,
                m_perturbations=10,
                lambda_=1.,
                use_sample=False,
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
        @param verbose, Boolean, whether present attack information or not
        """
        if x is None and x.shape[0] <= 0:
            return []
        adv_x = x.detach().clone().to(torch.float)
        self.lambda_ = lambda_
        padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        model.eval()
        for iter_i in range(m_perturbations):
            if use_sample and iter_i == 0:
                adv_x = rand_x(adv_x, is_sample=True)
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            hidden, logit = model.forward(var_adv_x, adj)
            loss, done = self.get_losses(model, logit, label, hidden)
            if torch.all(done):
                break
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0].data

            # filtering un-considered graphs & positions
            grad = grad * padding_mask
            grad4insertion = (grad > 0) * grad * (adv_x < 0.5)

            grad4ins_ = grad4insertion.reshape(x.shape[0], -1)
            _, pos = torch.max(grad4ins_, dim=-1)
            perturbation = F.one_hot(pos, num_classes=grad4ins_.shape[-1]).float().reshape(x.shape)
            adv_x = torch.clamp(adv_x + perturbation, min=0., max=1.)

            if verbose:
                print(
                    f"\n Iteration {iter_i}: the accuracy is {(logit.argmax(1) == 1.).sum().item() / adv_x.size()[0] * 100:.3f}.")
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
