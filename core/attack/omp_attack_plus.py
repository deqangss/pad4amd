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
    @manipulation_z, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, kappa=10, manipulation_z=None, omega=None, device=None):
        super(OMPAP, self).__init__(True, kappa, manipulation_z, omega, device)

    def perturb(self, model, node, adj=None, label=None,
                n_perturbations=10,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                verbose=False):
        assert 0 < min_lambda_ <= max_lambda_
        self.lambda_ = min_lambda_

        # self.perturb(model, node, n_perturbations, s)
        adv_node = node.detach().clone()
        while self.lambda_ <= max_lambda_:
            latent_rpst, logit = model.forward(adv_node, adj)
            _, done = self.get_losses(model, logit, label, latent_rpst)
            if torch.all(done):
                return adv_node

            adv_adj = None if adj is None else adv_adj[~done]
            pert_x = super(OMPAP, self).perturb(model, adv_node[~done], adv_adj, label[~done],
                                                n_perturbations,
                                                self.lambda_,
                                                step_length=1.,
                                                verbose=False
                                                )
            adv_node[~done] = pert_x
            self.lambda_ *= 10.
            if verbose:
                logger.info(f"Ompa attack: attack effectiveness {done.sum().item()/node.size()[0]} with lambda {self.lambda_}.")
        return adv_node
