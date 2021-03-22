"""
enhance 'omp_attack': (i) exponential search for looking for lambda; (2) change adversarial loss
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
    @param is_attacker, play the role of attacker (adversarial training indicates the defender rather than the attacker)
    @manipulation_z, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=False, kappa=10, manipulation_z=None, omega=None, device=None):
        super(OMPAP, self).__init__(is_attacker, kappa, manipulation_z, omega, device)

    def perturb(self, model, x, adj=None, label=None,
                m=10,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                granularity=10.,
                stop=True,
                verbose=False):
        assert 0 < min_lambda_ <= max_lambda_
        self.lambda_ = min_lambda_

        adv_x = x.detach().clone()
        while self.lambda_ <= max_lambda_:
            hidden, logit = model.forward(adv_x, adj)
            _, done = self.get_losses(model, logit, label, hidden)
            if verbose:
                logger.info(
                    f"Ompa attack: attack effectiveness {done.sum().item() / x.size()[0]} with lambda {self.lambda_}.")
            if torch.all(done):
                return adv_x

            adv_adj = None if adj is None else adv_adj[~done]
            pert_x = super(OMPAP, self).perturb(model, adv_x[~done], adv_adj, label[~done],
                                                m,
                                                self.lambda_,
                                                step_length=1.,
                                                stop=stop,
                                                verbose=False
                                                )
            adv_x[~done] = pert_x
            self.lambda_ *= granularity
        return adv_x
