"""
enhance 'omp_attack': (i) exponential search for looking for lambda; (2) change adversarial loss
"""
import torch
import numpy as np

from core.attack import OMPA
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.ompa_plus')
logger.addHandler(ErrorHandler)


class OMPAP(OMPA):
    """
    Orthogonal matching pursuit attack

    Parameters
    ---------
    @manipulation_z, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, manipulation_z=None, omega=None, device=None):
        super(OMPAP, self).__init__(manipulation_z, omega, device)

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
                _, done = self.get_losses(model, logit, label, hidden)
            if verbose:
                logger.info(f"Ompa attack: attack effectiveness {done.sum().item() / x.size()[0]} with lambda {self.lambda_}.")
            if torch.all(done):
                break
            # adv_x[~done] = x[~done]  # recompute the perturbation under other penalty factors
            adv_adj = None if adj is None else adv_adj[~done]
            pert_x = super(OMPAP, self).perturb(model, adv_x[~done], adv_adj, label[~done],
                                                m,
                                                self.lambda_,
                                                step_length=1.,
                                                clone=False,
                                                verbose=False
                                                )
            adv_x[~done] = pert_x
            self.lambda_ *= base
        with torch.no_grad():
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_losses(model, logit, label, hidden)
            if verbose:
                logger.info(f"Ompa: attack effectiveness {done.sum().item() / x.size()[0]}.")
        return adv_x
