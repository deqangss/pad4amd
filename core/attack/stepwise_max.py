import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, round_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.stepwise_max')
logger.addHandler(ErrorHandler)
EXP_OVER_FLOW = 1e-120


class StepwiseMax(BaseAttack):
    """
    Stepwise max attack (mixture of pgd l1, pgd l2, pgd linf)

    Parameters
    ---------
    @param use_random, Boolean,  whether use random start point
    @param rounding_threshold, float, a threshold for rounding real scalars
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, use_random=False, rounding_threshold=0.5,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(StepwiseMax, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.use_random = use_random
        assert 0 < rounding_threshold < 1
        self.round_threshold = rounding_threshold
        self.lambda_ = 1.

    def perturb(self, model, x, label=None,
                steps=100,
                sl_l1=1.,
                sl_l2=1.,
                sl_linf=0.01,
                min_lambda_=1e-5,
                max_lambda_=1e5,
                base=10.,
                verbose=False):
        """
        enhance attack
        """
        assert 0 < min_lambda_ <= max_lambda_
        assert steps >= 0 and 1 >= sl_l1 > 0 and sl_l2 >= 0 and sl_linf >= 0
        model.eval()
        if hasattr(model, 'forward_g'):
            self.lambda_ = min_lambda_
        else:
            self.lambda_ = max_lambda_
        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))

        adv_x = x.detach().clone()
        while self.lambda_ <= max_lambda_:
            pert_x_cont = None
            prev_done = None
            for i, mini_step in enumerate(range(steps)):
                with torch.no_grad():
                    if i == 0 and self.use_random:
                        adv_x = get_x0(adv_x, rounding_threshold=self.round_threshold, is_sample=True)
                    _, done = self.get_loss(model, adv_x, label, self.lambda_)
                if torch.all(done):
                    break
                if i == 0:
                    adv_x[~done] = x[~done]  # recompute the perturbation under other penalty factors
                    prev_done = done
                else:
                    adv_x[~done] = pert_x_cont[~done[~prev_done]]
                    prev_done = done

                num_sample_red = torch.sum(~done).item()
                pert_x_linf, pert_x_l2, pert_x_l1 = self._perturb(model, adv_x[~done], label[~done],
                                                                  sl_l1,
                                                                  sl_l2,
                                                                  sl_linf,
                                                                  lambda_=self.lambda_
                                                                  )
                with torch.no_grad():
                    pertb_x_list = [pert_x_linf, pert_x_l2, pert_x_l1]
                    n_attacks = len(pertb_x_list)
                    pertbx = torch.vstack(pertb_x_list)
                    label_ext = torch.cat([label[~done]] * n_attacks)
                    scores = self.get_scores(model, pertbx, label_ext)
                    pertbx = pertbx.reshape(n_attacks, num_sample_red, *red_n).permute([1, 0, *red_ind])
                    scores = scores.reshape(n_attacks, num_sample_red).permute(1, 0)
                    _, s_idx = scores.max(dim=-1)
                    pert_x_cont = pertbx[torch.arange(num_sample_red), s_idx]
                    if self.is_attacker:
                        adv_x[~done] = round_x(pert_x_cont, self.round_threshold)
                    else:
                        adv_x[~done] = pert_x_cont
            self.lambda_ *= base
            if not self.check_lambda(model):
                break
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if verbose:
                logger.info(f"step-wise max: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")
        return adv_x

    def _perturb(self, model, x, label=None,
                 step_length_l1=1.,
                 step_length_l2=1.,
                 step_length_linf=0.01,
                 lambda_=1.,
                 ):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param step_length_l1: float value in [0,1], the step length in each iteration
        @param step_length_l2: float, the step length in each iteration
        @param step_length_linf: float, the step length in each iteration
        @param lambda_, float, penalty factor
        """
        if x is None or x.shape[0] <= 0:
            return []
        self.lambda_ = lambda_
        assert 0 <= step_length_l1 <= 1, "Expected a real-value in [0,1], but got {}".format(step_length_l1)
        model.eval()
        adv_x = x.detach()
        var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
        loss, done = self.get_loss(model, var_adv_x, label, self.lambda_)
        grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0].data

        # look for allowable position, because only '1--> -' and '0 --> +' are permitted
        # api insertion
        pos_insertion = (adv_x <= 0.5) * 1 * (adv_x >= 0.)
        grad4insertion = (grad > 0) * pos_insertion * grad
        # api removal
        pos_removal = (adv_x > 0.5) * 1
        grad4removal = (grad <= 0) * (pos_removal & self.manipulation_x) * grad
        if self.is_attacker:
            #     2.1 cope with the interdependent apis
            checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
            grad4removal[:, self.api_flag] += torch.sum(grad * checking_nonexist_api, dim=-1, keepdim=True)
        grad = grad4removal + grad4insertion

        # linf
        perturbation_linf = torch.sign(grad)
        if self.is_attacker:
            perturbation_linf += (torch.any(perturbation_linf[:, self.api_flag] < 0, dim=-1,
                                            keepdim=True) * checking_nonexist_api)
        adv_x_linf = torch.clamp(adv_x + step_length_linf * perturbation_linf, min=0., max=1.)
        # l2
        l2norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
        perturbation_l2 = torch.minimum(
            torch.tensor(1., dtype=adv_x.dtype, device=adv_x.device),
            grad / l2norm
        )
        perturbation_l2 = torch.where(torch.isnan(perturbation_l2), 0., perturbation_l2)
        if self.is_attacker:
            min_val = torch.amin(perturbation_l2, dim=-1, keepdim=True).clamp_(max=0.)
            perturbation_l2 += (torch.any(perturbation_l2[:, self.api_flag] < 0, dim=-1,
                                          keepdim=True) * torch.abs(min_val) * checking_nonexist_api)
        adv_x_l2 = torch.clamp(adv_x + step_length_l2 * perturbation_l2, min=0., max=1.)
        # l1
        val, idx = torch.abs(grad).topk(int(1. / step_length_l1), dim=-1)
        perturbation_l1 = F.one_hot(idx, num_classes=adv_x.shape[-1]).sum(dim=1)
        perturbation_l1 = perturbation_linf * perturbation_l1
        if self.is_attacker:
            perturbation_l1 += (
                    torch.any(perturbation_l1[:, self.api_flag] < 0, dim=-1, keepdim=True) * checking_nonexist_api)
        adv_x_l1 = torch.clamp(adv_x + step_length_l1 * perturbation_l1, min=0., max=1.)
        return adv_x_linf, adv_x_l2, adv_x_l1