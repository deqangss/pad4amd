import torch
import torch.nn.functional as F
import numpy as np

from core.attack.base_attack import BaseAttack
from tools.utils import round_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.gdkde')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class GDKDE(BaseAttack):
    """
    a variant of gradient descent with kernel density estimation: we calculate the density estimation upon the hidden
    space and perturb the feature in the direction of l2 norm based gradients

    Parameters
    ---------
    @param ben_hidden: torch.Tensor, hidden representation of benign files on the hidden space
    @param bandwidth: float, variance of gaussian distribution
    @param kappa, float, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, ben_hidden=None, bandwidth=20.,
                 is_attacker=True, kappa=1., manipulation_x=None, omega=None, device=None):
        super(GDKDE, self).__init__(is_attacker, kappa, manipulation_x, omega, device)
        self.ben_hidden = ben_hidden
        self.bandwidth = bandwidth
        self.lambda_ = 1.
        if isinstance(self.ben_hidden, torch.Tensor):
            pass
        elif isinstance(self.ben_hidden, np.ndarray):
            self.ben_hidden = torch.tensor(self.ben_hidden, device=device)
        else:
            raise TypeError

    def _perturb(self, model, x, adj=None, label=None,
                 steps=10,
                 step_length=1.,
                 lambda_=1.,
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
        @param step_length: float, the step length in each iteration
        @param lambda_, float, penalty factor
        @param verbose, Boolean, whether present attack information or not
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x
        self.lambda_ = lambda_
        self.padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        model.eval()
        for t in range(steps):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            hidden, logit = model.forward(var_adv_x, adj)
            loss, done = self.get_loss(model, logit, label, hidden)
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0]
            perturbation = self.get_perturbation(grad, x, adv_x)
            # avoid to perturb the examples that are successful to evade the victim
            adv_x = torch.clamp(adv_x + perturbation * step_length, min=0., max=1.)
        return round_x(adv_x)

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
        if model.k > 0:
            logger.warning("The attack leads to dense graph and trigger the issue of out of memory.")
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
                                   steps,
                                   step_length,
                                   lambda_=self.lambda_,
                                   verbose=False
                                   )
            adv_x[~done] = pert_x
            self.lambda_ *= base
            if not self.check_lambda(model):
                break
        with torch.no_grad():
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_loss(model, logit, label, hidden)
            if verbose:
                logger.info(f"gdkde: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3}%.")
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
        #    2.2 api removal
        pos_removal = (adv_features > 0.5) * 1
        # #     2.2.1 cope with the interdependent apis
        # checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
        # grad4removal = torch.sum(gradients * checking_nonexist_api, dim=-1, keepdim=True) + gradients
        # grad4removal *= (grad4removal < 0) * (pos_removal & self.manipulation_x)
        grad4removal = (gradients < 0) * (pos_removal & self.manipulation_x) * gradients
        gradients = grad4removal + grad4insertion

        # 3. normalize gradient in the direction of l2 norm
        l2norm = torch.sqrt(torch.max(div_zero_overflow, torch.sum(gradients ** 2, dim=red_ind, keepdim=True)))
        perturbation = torch.minimum(
            torch.tensor(1., dtype=features.dtype, device=features.device),
            gradients / l2norm
        )
        return perturbation

    def get_loss(self, model, logit, label, hidden):
        ce = F.cross_entropy(logit, label, reduction='none')
        y_pred = logit.argmax(1)
        square = torch.sum(torch.square(self.ben_hidden.unsqueeze(dim=0) - hidden.unsqueeze(dim=1)), dim=-1)
        kde = torch.mean(torch.exp(-square / self.bandwidth), dim=-1)
        loss_no_reduction = ce + 1000 * kde
        if 'forward_g' in type(model).__dict__.keys():
            de = model.forward_g(hidden, y_pred)
            tau = model.get_tau_sample_wise(y_pred)
            if self.is_attacker:
                loss_no_reduction += self.lambda_ * (torch.clamp(
                    torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW), max=self.kappa))
            else:
                loss_no_reduction += self.lambda_ * \
                    torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW)
            done = (y_pred == 0.) & (de >= tau)
        else:
            done = y_pred == 0.
        return loss_no_reduction, done
