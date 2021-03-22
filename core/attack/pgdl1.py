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
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.pgdl1')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class PGDl1(BaseAttack):
    """
    Projected gradient descent (ascent) with gradients 'normalized' using l1 norm.
    By comparing BCA, the api removal is leveraged

    Parameters
    ---------
    @param kappa, attack confidence
    @param manipulation_z, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, kappa=10., manipulation_z=None, omega=None, device=None):
        super(PGDl1, self).__init__(manipulation_z, omega, device)
        self.kappa = kappa
        self.lambda_ = 1.

    def perturb(self, model, x, adj=None, label=None,
                m_perturbations=10,
                lambda_=1.,
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
        @param stop, Boolean, whether stop once evade victim successfully
        @param verbose, Boolean, whether present attack information or not
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x.detach().clone().to(torch.float)
        self.lambda_ = lambda_
        self.padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        model.eval()
        for t in range(m_perturbations):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            hidden, logit = model.forward(var_adv_x, adj)
            loss, done = self.get_losses(model, logit, label, hidden)
            if verbose:
                print(
                    f"\n Iteration {t}: the accuracy is {(logit.argmax(1) == 1.).sum().item() / adv_x.size()[0] * 100:.3f}.")
            if torch.all(done) and stop:
                break
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0].data
            perturbation, direction = self.get_perturbation(grad, x, adv_x)
            if stop:
                perturbation[done] = 0.
            adv_x = torch.clamp(adv_x + perturbation * direction, min=0., max=1.)
        return adv_x

    def perturb_ehs(self, model, x, adj=None, label=None,
                    m=10,
                    min_lambda_=1e-5,
                    max_lambda_=1e5,
                    granularity=10.,
                    stop=True,
                    verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        self.lambda_ = min_lambda_
        adv_x = x.clone()
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
                                  m,
                                  lambda_=self.lambda_,
                                  stop=stop,
                                  verbose=False
                                  )
            adv_x[~done] = pert_x
            self.lambda_ *= granularity
        return adv_x

    def get_perturbation(self, gradients, features, adv_features):
        # 1. mask paddings
        gradients = gradients * self.padding_mask

        # 2. look for allowable position, because only '1--> -' and '0 --> +' are permitted
        #    2.1 api insertion
        pos_insertion = (adv_features < 0.5) * 1
        grad4insertion = (gradients > 0) * pos_insertion * gradients
        #    2.2 api removal
        pos_removal = (adv_features >= 0.5) * 1
        #     2.2.1 cope with the interdependent apis
        checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
        grad4removal = torch.sum(gradients * checking_nonexist_api, dim=-1, keepdim=True) + gradients
        grad4removal *= (grad4removal < 0) * (pos_removal & self.manipulation_z)
        gradients = grad4removal + grad4insertion

        # 3. remove duplications
        un_mod = torch.abs(features - adv_features) <= 1e-6
        gradients = gradients * un_mod

        # 4. look for important position
        absolute_grad = torch.abs(gradients).reshape(features.shape[0], -1)
        _, position = torch.max(absolute_grad, dim=-1)
        perturbations = F.one_hot(position, num_classes=absolute_grad.shape[-1]).float()
        perturbations = perturbations.reshape(features.shape)
        directions = torch.sign(gradients) * (perturbations > 1e-6)

        # 5. tailor the interdependent apis
        perturbations += (torch.sum(directions, dim=-1, keepdim=True) < 0) * checking_nonexist_api
        directions += perturbations * self.omega
        return perturbations, directions

    def get_losses(self, model, logit, label, hidden=None):
        ce = F.cross_entropy(logit, label, reduction='none')
        y_pred = logit.argmax(1)
        if 'forward_g' in type(model).__dict__.keys():
            de = model.forward_g(hidden, y_pred)
            tau = model.get_tau_sample_wise(y_pred)
            loss_no_reduction = ce + self.lambda_ * (torch.clamp(
                torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW), max=self.kappa))
            done = (y_pred == 0.) & (de >= tau)
        else:
            loss_no_reduction = ce
            done = y_pred == 0.
        return loss_no_reduction, done
