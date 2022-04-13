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
import numpy as np

from core.attack.base_attack import BaseAttack
from tools.utils import get_x0, round_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.stepwise_max')
logger.addHandler(ErrorHandler)


class StepwiseMax(BaseAttack):
    """
    Stepwise max attack (mixture of pgd l1, pgd l2, pgd linf

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

    def _perturb(self, model, x, label=None,
                 steps=10,
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
        @param steps: Integer, maximum number of iterations
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
        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))
        adv_x = x.detach().clone()
        stop_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
        for t in range(steps):
            num_sample_red = n - torch.sum(stop_flag)
            if t == 0 and self.use_random:
                adv_x = get_x0(adv_x, rounding_threshold=self.round_threshold, is_sample=True)
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss, done = self.get_loss(model, var_adv_x, label, self.lambda_)
            grad = torch.autograd.grad(torch.mean(loss), var_adv_x)[0].detach().data
            pertbx_list = self.get_perturbation(grad, x, adv_x, step_length_l1, step_length_l2, step_length_linf)

            with torch.no_grad():
                n_attacks = len(pertbx_list)
                pertbx = torch.vstack(pertbx_list)
                label_ext = torch.cat([label] * n_attacks)
                scores, done = self.get_scores(model, pertbx, label_ext)
                pertbx = pertbx.reshape(n_attacks, num_sample_red, *red_n).permute([1, 0, *red_ind])
                scores = scores.reshape(n_attacks, num_sample_red).permute(1, 0)
                _, s_idx = scores.max(dim=-1)
                print(scores[:10])
                print(s_idx)
                adv_x = pertbx[torch.arange(num_sample_red), s_idx]
        return adv_x

    def perturb(self, model, x, label=None,
                steps=100,
                step_check=10,
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
        mini_steps = [step_check] * (steps // step_check)
        mini_steps = mini_steps + [steps % step_check] if steps % step_check != 0 else mini_steps

        adv_x = x.detach().clone().to(torch.double)
        while self.lambda_ <= max_lambda_:
            pert_x_cont = None
            prev_done = None
            for i, mini_step in enumerate(mini_steps):
                with torch.no_grad():
                    _, done = self.get_loss(model, adv_x, label, self.lambda_)
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
                                            sl_l1,
                                            sl_l2,
                                            sl_linf,
                                            lambda_=self.lambda_
                                            )
                adv_x[~done] = round_x(pert_x_cont, self.round_threshold)
            self.lambda_ *= base
            if not self.check_lambda(model):
                break
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if verbose:
                logger.info(f"step-wise max: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")
        return adv_x

    def get_perturbation(self, gradients, x, adv_x, sl_l1, sl_l2, sl_linf):
        # look for allowable position, because only '1--> -' and '0 --> +' are permitted
        # api insertion
        pos_insertion = (adv_x <= 0.5) * 1 * (adv_x >= 0.)
        grad4insertion = (gradients > 0) * pos_insertion * gradients
        # api removal
        pos_removal = (adv_x > 0.5) * 1
        grad4removal = (gradients <= 0) * (pos_removal & self.manipulation_x) * gradients
        if self.is_attacker:
            #     2.1 cope with the interdependent apis
            checking_nonexist_api = (pos_removal ^ self.omega) & self.omega
            grad4removal[:, self.api_flag] += torch.sum(gradients * checking_nonexist_api, dim=-1, keepdim=True)

        gradients = grad4removal + grad4insertion

        pertbx = []
        # norm
        #    linf norm
        perturbation_linf = torch.sign(gradients)
        if self.is_attacker:
            perturbation_linf += (torch.any(perturbation_linf[:, self.api_flag] < 0, dim=-1,
                                            keepdim=True) * checking_nonexist_api)
        perturbx_linf = torch.clamp(adv_x + sl_linf * perturbation_linf, min=0., max=1.)
        pertbx.append(perturbx_linf)
        #    l2 norm
        l2norm = torch.linalg.norm(gradients)
        perturbation_l2 = torch.minimum(
            torch.tensor(1., dtype=x.dtype, device=x.device),
            gradients / l2norm
        )
        if self.is_attacker:
            min_val = torch.amin(perturbation_l2, dim=-1, keepdim=True).clamp_(max=0.)
            perturbation_l2 += (torch.any(perturbation_l2[:, self.api_flag] < 0, dim=-1,
                                          keepdim=True) * torch.abs(min_val) * checking_nonexist_api)
        perturbx_l2 = torch.clamp(adv_x + sl_l2 * perturbation_l2, min=0., max=1.)
        pertbx.append(perturbx_l2)
        #    l1 norm
        k = int(1. / sl_l1)
        val, idx = torch.abs(gradients).topk(k, dim=-1)
        perturbation_l1 = F.one_hot(idx, num_classes=x.shape[-1]).sum(dim=1).double()
        perturbation_l1 = perturbation_linf * perturbation_l1
        perturbation_l1 += (torch.any(perturbation_l1[:, self.api_flag] < 0, dim=-1, keepdim=True) * checking_nonexist_api)
        perturbx_l1 = torch.clamp(adv_x + sl_l1 * perturbation_l1, min=0., max=1.)
        pertbx.append(perturbx_l1)
        return pertbx

    def get_scores(self, model, pertb_x, label):
        logits_f = model.forward_f(pertb_x)
        ce = F.cross_entropy(logits_f, label, reduction='none')
        y_pred = logits_f.argmax(1)
        if 'forward_g' in type(model).__dict__.keys() and (not self.oblivion):
            logits_g = model.forward_g(pertb_x)
            loss_no_reduction = ce + model.tau - logits_g
            done = (y_pred == 0.) & (logits_g <= model.tau)
        else:
            loss_no_reduction = ce
            done = y_pred == 0.
        return loss_no_reduction, done