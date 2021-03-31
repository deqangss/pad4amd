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
    def __init__(self, attack_list, varepsilon=1e-9, kappa=1., manipulation_x=None, omega=None, device=None):
        super(Max, self).__init__(kappa, manipulation_x, omega, device)
        assert len(attack_list) > 0, 'Expect one attack at least.'
        self.attack_list = attack_list
        self.varepsilon = varepsilon
        self.device = device

    def perturb(self, model, x, adj=None, label=None, steps=5, verbose=False):
        """
        perturb node features

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param adj: torch.FloatTensor or None, adjacency matrix (if not None, the shape is [number_of_graphs, batch_size, vocab_dim, vocab_dim])
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of iterations
        @param verbose: Boolean, print verbose log
        """
        if x is None or x.shape[0] <= 0:
            return []
        with torch.no_grad():
            hidden, logit = model.forward(x, adj)
            loss, done = self.get_loss(model, logit, label, hidden)
        worst_loss = loss
        n, red_n = x.size()[0], x.size()[1:]
        red_ind = list(range(2, len(x.size()) + 1))
        adv_x = x.detach().clone()
        stop_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
        for t in range(steps):
            num_sample_red = n - torch.sum(stop_flag)
            if num_sample_red <= 0:
                return adv_x

            adj_red = None if adj is None else adj[~stop_flag]
            label_red = label[~stop_flag]
            pertbx = []
            for attack in self.attack_list:
                assert 'perturb' in type(attack).__dict__.keys()
                if t > 0 and 'use_random' in attack.__dict__.keys():
                    attack.use_random = False
                adj = None if adj is None else adj[~stop_flag]
                pertbx.append(attack.perturb(model, adv_x[~stop_flag], adj_red, label_red))
            pertbx = torch.vstack(pertbx)

            with torch.no_grad():
                adj_red_ext = None if adj is None else torch.vstack([adj_red] * len(self.attack_list))
                label_red_ext = torch.cat([label_red] * len(self.attack_list))
                hidden, logit = model.forward(pertbx, adj_red_ext)
                loss, done = self.get_loss(model, logit, label_red_ext, hidden)
                loss = loss.reshape(len(self.attack_list), num_sample_red).permute(1, 0)
                done = done.reshape(len(self.attack_list), num_sample_red).permute(1, 0)
                success_flag = torch.any(done, dim=-1)
                done[~torch.any(done, dim=-1)] = 1  # for a sample, if no attacks evade the victim successful, all perturbed examples are reminded for future selection
                loss = (loss * done.to(torch.float)) + torch.min(loss) * (~done).to(torch.float)
                pertbx = pertbx.reshape(len(self.attack_list), num_sample_red, *red_n).permute([1, 0, *red_ind])
                _, indices = loss.max(dim=-1)
                adv_x[~stop_flag] = pertbx[torch.arange(num_sample_red), indices]
                a_loss = loss[torch.arange(num_sample_red), indices]
                worst_loss[~stop_flag] = a_loss
                stop_flag[~stop_flag] = (torch.abs(worst_loss[~stop_flag] - a_loss) < self.varepsilon) | success_flag

                if verbose:
                    hidden, logit = model.forward(adv_x, adj)
                    _, done = self.get_loss(model, logit, label, hidden)
                    logger.info(f"max: attack effectiveness {done.sum().item() / x.size()[0]}.")
        return adv_x

    def get_loss(self, model, logit, label, hidden=None):
        ce = F.cross_entropy(logit, label, reduction='none')
        y_pred = logit.argmax(1)
        if 'forward_g' in type(model).__dict__.keys():
            de = model.forward_g(hidden, y_pred)
            tau = model.get_tau_sample_wise(y_pred)
            loss_no_reduction = ce + torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW)
            # loss_no_reduction = ce + self.lambda_ * (de - model.tau)
            done = (y_pred == 0.) & (de >= tau)
        else:
            loss_no_reduction = ce
            done = y_pred == 0.
        return loss_no_reduction, done

