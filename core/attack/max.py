import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.max')
logger.addHandler(ErrorHandler)
EXP_OVER_FLOW = 1e-30


class Max(BaseAttack):
    """
    max attack: select results from several attacks iteratively

    Parameters
    --------
    @param attack_list: List, a list of instantiated attack object
    @param varepsilon: Float, a scaler for justifying the convergence
    """
    def __init__(self, attack_list, varepsilon=1e-9,
                 is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(Max, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        assert len(attack_list) > 0, 'Expect one attack at least.'
        self.attack_list = attack_list
        self.varepsilon = varepsilon
        self.device = device

    def perturb(self, model, x, label=None, steps_of_max=5, min_lambda_=1e-5, max_lambda_=1e5, verbose=False):
        """
        perturb node features

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param steps_of_max: Integer, maximum number of iterations
        @param min_lambda_, float, minimum value of penalty factor
        @param max_lambda_, float, maximum value of penalty factor
        @param verbose: Boolean, print verbose log
        """
        if x is None or x.shape[0] <= 0:
            return []
        model.eval()
        with torch.no_grad():
            loss, done = self.get_loss_without_lambda(model, x, label)
        pre_loss = loss
        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))
        adv_x = x.detach().clone()
        stop_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
        for t in range(steps_of_max):
            num_sample_red = n - torch.sum(stop_flag)
            if num_sample_red <= 0:
                break

            red_label = label[~stop_flag]
            pertbx = []
            for attack in self.attack_list:
                assert 'perturb' in type(attack).__dict__.keys()
                if t > 0 and 'use_random' in attack.__dict__.keys():
                    attack.use_random = False
                pertbx.append(attack.perturb(model=model, x=adv_x[~stop_flag], label=red_label,
                                             min_lambda_=min_lambda_,
                                             max_lambda_=max_lambda_))
            pertbx = torch.vstack(pertbx)

            with torch.no_grad():
                red_label_ext = torch.cat([red_label] * len(self.attack_list))
                loss, done = self.get_loss_without_lambda(model, pertbx, red_label_ext)
                loss = loss.reshape(len(self.attack_list), num_sample_red).permute(1, 0)
                done = done.reshape(len(self.attack_list), num_sample_red).permute(1, 0)
                success_flag = torch.any(done, dim=-1)
                # for a sample, if there is at least one successful attack, we will select the one with maximum loss;
                # while if no attacks evade the victim successful, all perturbed examples are reminded for selection
                done[~torch.any(done, dim=-1)] = 1
                loss = (loss * done.to(torch.float)) + torch.min(loss) * (~done).to(torch.float)
                pertbx = pertbx.reshape(len(self.attack_list), num_sample_red, *red_n).permute([1, 0, *red_ind])
                _, indices = loss.max(dim=-1)
                adv_x[~stop_flag] = pertbx[torch.arange(num_sample_red), indices]
                a_loss = loss[torch.arange(num_sample_red), indices]
                pre_stop_flag = stop_flag.clone()
                stop_flag[~stop_flag] = (torch.abs(pre_loss[~stop_flag] - a_loss) < self.varepsilon) | success_flag
                pre_loss[~pre_stop_flag] = a_loss
        if verbose:
            with torch.no_grad():
                _, done = self.get_loss_without_lambda(model, adv_x, label)
                logger.info(f"max: attack effectiveness {done.sum().item() / x.size()[0] * 100}%.")
        return adv_x

    def get_loss_without_lambda(self, model, pertb_x, label):
        logits_f = model.forward_f(pertb_x)
        ce = F.cross_entropy(logits_f, label, reduction='none')
        y_pred = logits_f.argmax(1)
        if 'forward_g' in type(model).__dict__.keys() and (not self.oblivion):
            logits_g = model.forward_g(pertb_x)
            tau = model.get_tau_sample_wise()
            loss_no_reduction = tau - logits_g
            done = (y_pred == 0.) & (logits_g <= tau)
        else:
            loss_no_reduction = ce
            done = y_pred == 0.
        return loss_no_reduction, done

