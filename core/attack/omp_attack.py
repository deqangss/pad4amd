from tqdm import tqdm

import torch
import torch.nn.functional as F

from tools import utils
from core.attack.base_attack import BaseAttack


class OMPA(BaseAttack):
    """
    Orthogonal matching pursuit attack

    Parameters
    ---------
    @param lambda_, float, penalty factor
    @param n_perturbations: Integer, number of perturbations
    @manipulation_z, manipulations
    @param omega, list of 4 sets, each set contains the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, lambda_=1., n_perturbations=10, manipulation_z=None, omega=None, device=None):
        super(OMPA, self).__init__(n_perturbations, manipulation_z, omega, device)
        self.lambda_ = lambda_

    def perturb(self, model, node, adj=None, label=None, step_length=1., verbose=False):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param node: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param adj: torch.FloatTensor or None, adjacency matrix (if not None, the shape is [number_of_graphs, batch_size, vocab_dim, vocab_dim])
        @param label: torch.LongTensor, ground truth labels
        @param step_length: Float, value is in the range of (0,1]
        """
        if node is None and node.shape[0] == 0:
            return []
        assert 0 < step_length <= 1.
        node, adj, label = utils.to_device(node, adj, label, self.device)
        adv_node = node.detach().clone().to(torch.float)
        model.eval()
        self.padding_mask = torch.sum(node, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        for iter_i in tqdm(range(self.n_perturbations)):
            var_adv_node = torch.autograd.Variable(adv_node, requires_grad=True)
            rpst, logit = model.forward(var_adv_node, adj)
            adv_loss = self.get_losses(model, logit, label, rpst)
            grads = torch.autograd.grad(torch.mean(adv_loss), var_adv_node)[0]
            perturbation, direction = self.get_perturbation(node, adv_node, grads)
            # cope with step length < 1.
            if 0 < step_length <= .5:
                with torch.no_grad():
                    steps = int(1 / step_length)
                    b, k, v = node.size()
                    perturbations = torch.stack([perturbation * direction * gamma for gamma in torch.linspace(step_length, 1., steps)], dim=0)
                    _adv_node = torch.clip((adv_node.unsqueeze(dim=0) + perturbations).permute(1, 0, 2, 3).reshape(b * steps, k, v), min=0, max=1)
                    if adj is not None:
                        adj = torch.repeat_interleave(adj, repeats=steps, dim=0)
                    _rpst, _logit = model.forward(_adv_node, adj)
                    _adv_loss = self.get_losses(model, _logit, torch.repeat_interleave(label, steps), _rpst)
                    _, _worst_pos = torch.max(_adv_loss.reshape(b, steps), dim=1)
                    adv_node = _adv_node.reshape(b, steps, k, v)[torch.arange(b), _worst_pos]
            else:
                adv_node = torch.clip(adv_node + perturbation * direction, min=0., max=1.)
            if verbose:
                print(f"\n Iteration {iter_i}: the accuracy is {(logit.argmax(1) == 1.).sum().item() / adv_node.size()[0]*100:.3f} with the loss {torch.mean(adv_loss).detach().cpu().numpy():.5f}.")
        return adv_node

    def get_losses(self, model, logit, label, representation=None):
        ce = F.cross_entropy(logit, label, reduction='none')
        g = model.forward_g(representation)
        return ce + self.lambda_ * (model.tau - g)

    def get_perturbation(self, features, adv_features, gradients):
        # 1. mask paddings
        gradients = gradients * self.padding_mask

        # 2. avoid the filled position, because only '1--> -' and '0 --> +' are permitted
        pos_insertion = (adv_features < 0.5) * 1
        grad4insertion = (gradients > 0) * pos_insertion * gradients
        pos_removal = (adv_features >= 0.5) * 1
        #   2.1 cope with the interdependent apis
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
