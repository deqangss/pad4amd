"""
enhance 'omp_attack': (i) exponential search for looking for lambda;
"""
import torch

from core.attack import OMPA
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.ompa_plus')
logger.addHandler(ErrorHandler)


class OMPAP(OMPA):
    """
    Orthogonal matching pursuit attack

    Parameters
    ---------
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(OMPAP, self).__init__(True, oblivion, kappa, manipulation_x, omega, device)

    def perturb(self, model, x, adj=None, label=None,
                m=10,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                base=10.,
                verbose=False):
        assert 0 < min_lambda_ <= max_lambda_
        adv_x = x.detach().clone().to(torch.float)
        self.lambda_ = min_lambda_
        while self.lambda_ <= max_lambda_:
            with torch.no_grad():
                hidden, logit = model.forward(adv_x, adj)
                _, done = self.get_loss(model, logit, label, hidden)
            if verbose:
                logger.info(f"Ompa attack: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3}% with lambda {self.lambda_}.")
            if torch.all(done):
                break
            adv_x[~done] = x[~done]  # recompute the perturbation under other penalty factors
            adv_adj = None if adj is None else adj[~done]
            pert_x = super(OMPAP, self).perturb(model, adv_x[~done], adv_adj, label[~done],
                                                m,
                                                self.lambda_,
                                                step_length=1.,
                                                clone=False,
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
                logger.info(f"Ompa: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3}%.")
        return adv_x
