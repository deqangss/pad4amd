import torch
import torch.nn.functional as F
import numpy as np

from core.attack.base_attack import BaseAttack
from tools.utils import round_x, get_x0
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
    @param benign_feat: torch.Tensor, representation of benign files on the feature space
    @param bandwidth: float, variance of gaussian distribution
    @param penalty_factor: float, penalty factor on kernel density estimation
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, float, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, benign_feat=None, bandwidth=20., penalty_factor=1000.,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(GDKDE, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.benign_feat = benign_feat
        self.bandwidth = bandwidth
        self.penalty_factor = penalty_factor
        self.lambda_ = 1.
        if isinstance(self.benign_feat, torch.Tensor):
            pass
        elif isinstance(self.benign_feat, np.ndarray):
            self.benign_feat = torch.tensor(self.benign_feat, device=device).double()
        else:
            raise TypeError

    def _perturb(self, model, x, label=None,
                 steps=10,
                 step_length=1.,
                 lambda_=1.):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.DoubleTensor, feature vectors with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of iterations
        @param step_length: float, the step length in each iteration
        @param lambda_, float, penalty factor
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x
        self.lambda_ = lambda_
        self.padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        model.eval()
        for t in range(steps):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label)
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0]
            perturbation = self.get_perturbation(grad, x, adv_x)
            # avoid to perturb the examples that are successful to evade the victim
            adv_x = torch.clamp(adv_x + perturbation * step_length, min=0., max=1.)
        return adv_x

    def perturb(self, model, x, label=None,
                steps=10,
                step_length=1.,
                step_check=10,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                base=10.,
                verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        assert steps >= 0 and step_check > 0 and step_length >= 0
        model.eval()

        if hasattr(model, 'forward_g'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_

        mini_steps = [step_check] * (steps // step_check)
        mini_steps = mini_steps + [steps % step_check] if steps % step_check != 0 else mini_steps

        adv_x = x.detach().clone().to(torch.double)
        adv_x = get_x0(adv_x, rounding_threshold=0.5, is_sample=True)
        while self.lambda_ <= max_lambda_:
            pert_x_cont = None
            prev_done = None
            for i, mini_step in enumerate(mini_steps):
                _, done = self.get_loss(model, adv_x, label)
                if torch.all(done):
                    break
                if i == 0:
                    adv_x[~done] = x[~done]  # recompute the perturbation under other penalty factors
                    prev_done = done
                else:
                    adv_x[~done] = pert_x_cont[~done[~prev_done]]
                    prev_done = done
                pert_x_cont = self._perturb(model, adv_x[~done], label[~done],
                                            mini_step,
                                            step_length,
                                            lambda_=self.lambda_
                                            )
                # round
                adv_x[~done] = pert_x_cont.round()

            self.lambda_ *= base
            if not self.check_lambda(model):
                break
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label)
            if verbose:
                logger.info(f"gdkde: attack effectiveness {done.sum().item() / x.size()[0] * 100:.3}%.")
        return adv_x

    def get_perturbation(self, gradients, features, adv_features):
        div_zero_overflow = torch.tensor(1e-30, dtype=gradients.dtype, device=gradients.device)
        red_ind = list(range(1, len(features.size())))

        # 1. look for allowable position, because only '1--> -' and '0 --> +' are permitted
        #    1.1 api insertion
        pos_insertion = (adv_features <= 0.5) * 1 * (adv_features >= 0.)
        grad4insertion = (gradients > 0) * pos_insertion * gradients
        #    1.2 api removal
        pos_removal = (adv_features > 0.5) * 1
        # #     2.2.1 cope with the interdependent apis
        # checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
        # grad4removal = torch.sum(gradients * checking_nonexist_api, dim=-1, keepdim=True) + gradients
        # grad4removal *= (grad4removal < 0) * (pos_removal & self.manipulation_x)
        grad4removal = (gradients < 0) * (pos_removal & self.manipulation_x) * gradients
        gradients = grad4removal + grad4insertion

        # 2. normalize gradient in the direction of l2 norm
        l2norm = torch.sqrt(torch.max(div_zero_overflow, torch.sum(gradients ** 2, dim=red_ind, keepdim=True)))
        perturbation = torch.minimum(
            torch.tensor(1., dtype=features.dtype, device=features.device),
            gradients / l2norm
        )
        return perturbation

    def get_loss(self, model, adv_x, label):
        logits_f = model.forward_f(adv_x)
        ce = F.cross_entropy(logits_f, label, reduction='none')
        print(ce[:10])
        y_pred = logits_f.argmax(1)
        square = torch.sum(torch.square(self.benign_feat.unsqueeze(dim=0) - adv_x.unsqueeze(dim=1)),
                           dim=-1)
        kde = torch.mean(torch.exp(-square / self.bandwidth), dim=-1)
        loss_no_reduction = ce + self.penalty_factor * kde
        if hasattr(model, 'forward_g') and (not self.oblivion):
            logits_g = model.forward_g(adv_x)
            tau = model.get_tau_sample_wise(y_pred)
            if self.is_attacker:
                loss_no_reduction += self.lambda_ * (torch.clamp(tau - logits_g, max=self.kappa))
            else:
                loss_no_reduction += self.lambda_ * (tau - logits_g)
            done = (y_pred == 0.) & (logits_g <= tau)
        else:
            done = y_pred == 0.
        return loss_no_reduction, done
