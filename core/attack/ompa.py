import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack
from tools.utils import rand_x
EXP_OVER_FLOW = -30


class OMPA(BaseAttack):
    """
    Orthogonal matching pursuit attack

    Parameters
    ---------
    @param is_attacker, play the role of attack or not
    @param kappa, attack confidence
    @param manipulation_z, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=False, kappa=10., manipulation_z=None, omega=None, device=None):
        super(OMPA, self).__init__(manipulation_z, omega, device)
        self.is_attacker = is_attacker
        assert kappa > 0.
        self.kappa = kappa
        self.lambda_ = 1.

    def perturb(self, model, node, adj=None, label=None,
                m_perturbations=10,
                lambda_=1.,
                step_length=1.,
                stop=True,
                verbose=False):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param node: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param adj: torch.FloatTensor or None, adjacency matrix (if not None, the shape is [number_of_graphs, batch_size, vocab_dim, vocab_dim])
        @param label: torch.LongTensor, ground truth labels
        @param m_perturbations: Integer, maximum number of perturbations
        @param lambda_, float, penalty factor
        @param step_length: Float, value is in the range of (0,1]
        @param stop: Boolean, whether stop once evade victim successfully
        @param verbose, Boolean, whether present attack information or not
        """
        if node is None and node.shape[0] == 0:
            return []
        assert 0 < step_length <= 1.
        # node, adj, label = utils.to_device(node, adj, label, self.device)
        adv_node = node.detach().clone().to(torch.float)
        self.lambda_ = lambda_
        self.padding_mask = torch.sum(node, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        model.eval()
        for t in range(m_perturbations):
            var_adv_node = torch.autograd.Variable(adv_node, requires_grad=True)
            hidden, logit = model.forward(var_adv_node, adj)
            adv_loss, done = self.get_losses(model, logit, label, hidden)
            if torch.all(done) and stop:
                break
            grad = torch.autograd.grad(torch.mean(adv_loss), var_adv_node)[0].data
            perturbation, direction = self.get_perturbation(grad, node, adv_node)
            if stop:
                perturbation[done] = 0.
            # cope with step length < 1.
            if 0 < step_length <= .5:
                with torch.no_grad():
                    steps = int(1 / step_length)
                    b, k, v = node.size()
                    perturbations = torch.stack(
                        [perturbation * direction * gamma for gamma in torch.linspace(step_length, 1., steps)], dim=0)
                    adv_node_expanded = torch.clip(
                        (adv_node.unsqueeze(dim=0) + perturbations).permute(1, 0, 2, 3).reshape(b * steps, k, v), min=0,
                        max=1)
                    if adj is not None:
                        adj = torch.repeat_interleave(adj, repeats=steps, dim=0)
                    rpst_, logit_ = model.forward(adv_node_expanded, adj)
                    adv_loss_ = self.get_losses(model, logit_, torch.repeat_interleave(label, steps), rpst_)
                    _, _worst_pos = torch.max(adv_loss_.reshape(b, steps), dim=1)
                    adv_node = adv_node_expanded.reshape(b, steps, k, v)[torch.arange(b), _worst_pos]
            else:
                adv_node = torch.clip(adv_node + perturbation * direction, min=0., max=1.)
        return adv_node

    def get_losses(self, model, logit, label, hidden=None):
        ce = F.cross_entropy(logit, label, reduction='none')
        if 'forward_g' in type(model).__dict__.keys():
            de = model.forward_g(hidden, logit.argmax(1))
            tau = model.get_tau_sample_wise(logit.argmax(1))
            if not self.is_attacker:
                loss_no_reduction = ce + self.lambda_ * \
                                    (torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW))
            else:
                loss_no_reduction = ce + \
                                    self.lambda_ * (torch.clamp(
                    torch.log(de + EXP_OVER_FLOW) - torch.log(tau + EXP_OVER_FLOW), max=self.kappa))
            # loss_no_reduction = ce + self.lambda_ * (de - model.tau)
            # print('cross-entropy:', ce)
            # print('density-estimation:', de)
            done = (logit.argmax(1) == 0.) & (de >= tau)
        else:
            loss_no_reduction = ce
            done = logit.argmax(1) == 0.
        return loss_no_reduction, done

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
        directions = torch.sign(gradients) * perturbations

        # 5. tailor the interdependent apis
        perturbations += (torch.sum(directions, dim=-1, keepdim=True) < 0) * checking_nonexist_api
        directions += perturbations * self.omega
        return perturbations, directions
