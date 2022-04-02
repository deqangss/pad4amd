import torch
import torch.nn.functional as F

from core.attack.base_attack import BaseAttack

EXP_OVER_FLOW = 1e-30


class OMPA(BaseAttack):
    """
    Orthogonal matching pursuit attack

    Parameters
    ---------
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param oblivion, Boolean, whether know the adversary indicator or not
    @param kappa, float, attack confidence
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, is_attacker=True, oblivion=False, kappa=1., manipulation_x=None, omega=None, device=None):
        super(OMPA, self).__init__(is_attacker, oblivion, kappa, manipulation_x, omega, device)
        self.is_attacker = is_attacker
        self.lambda_ = 1.

    def perturb(self, model, x, label=None,
                m=10,
                lambda_=1.,
                step_length=1.,
                clone=True,
                verbose=False):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param m: Integer, maximum number of perturbations
        @param lambda_, float, penalty factor
        @param step_length: Float, value is in the range of (0,1]
        @param clone: Boolean, whether clone the node feature
        @param verbose, Boolean, whether present attack information or not
        """
        if x is None or x.shape[0] == 0:
            return []
        assert 0 < step_length <= 1.
        # node, adj, label = utils.to_device(x, adj, label, self.device)
        if clone:
            adv_x = x.detach().clone().to(torch.float)
        else:
            adv_x = x
        self.lambda_ = lambda_
        model.eval()
        for t in range(m):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            adv_loss, done = self.get_loss(model, var_adv_x, label)
            if torch.all(done):
                break
            grad = torch.autograd.grad(torch.mean(adv_loss), var_adv_x)[0]
            perturbation, direction = self.get_perturbation(grad, x, adv_x)
            # avoid to perturb the examples that are successful to evade the victim
            # note: this can decrease the transferability of adversarial examples
            perturbation[done] = 0.
            # cope with step length < 1.
            if 0 < step_length <= .5 and (not self.is_attacker):
                with torch.no_grad():
                    steps = int(1 / step_length)
                    b, dim = x.size()
                    perturbations = torch.stack(
                        [perturbation * direction * gamma for gamma in torch.linspace(step_length, 1., steps)], dim=0)
                    adv_x_expanded = torch.clip(
                        (adv_x.unsqueeze(dim=0) + perturbations).permute(1, 0, 2).reshape(b * steps, dim),
                        min=0,
                        max=1)
                    logits_ = model.forward(adv_x_expanded)
                    adv_loss_ = self.get_loss(model, logits_, torch.cat([label] * steps, dim=0))
                    _, worst_pos = torch.max(adv_loss_.reshape(b, steps), dim=1)
                    adv_x = adv_x_expanded.reshape(b, steps, dim)[torch.arange(b), worst_pos]
            else:
                adv_x = torch.clip(adv_x + perturbation * direction, min=0., max=1.)
        return adv_x

    def get_perturbation(self, gradients, features, adv_features):
        # 1. look for allowable position, because only '1--> -' and '0 --> +' are permitted
        #    1.1 api insertion
        pos_insertion = (adv_features <= 0.5) * 1
        grad4insertion = (gradients > 0) * pos_insertion * gradients  # owing to gradient ascent
        # 2. api removal
        pos_removal = (adv_features > 0.5) * 1
        if self.is_attacker:
            #     2.1 cope with the interdependent apis (note: the following is application-specific)
            checking_nonexist_api = (pos_removal ^ self.omega) & self.omega  # broadcasting
            grad4removal = torch.sum(gradients * checking_nonexist_api, dim=-1, keepdim=True) + gradients
            grad4removal *= (grad4removal < 0) * (pos_removal & self.manipulation_x)
        else:
            grad4removal = (gradients < 0) * (pos_removal & self.manipulation_x) * gradients
        gradients = grad4removal + grad4insertion

        # 3. remove duplications (i.e., neglect the positions whose values have been modified previously.)
        un_mod = torch.abs(features - adv_features) <= 1e-6
        gradients = gradients * un_mod

        # 4. look for important position
        absolute_grad = torch.abs(gradients).reshape(features.shape[0], -1)
        _, position = torch.max(absolute_grad, dim=-1)
        perturbations = F.one_hot(position, num_classes=absolute_grad.shape[-1]).float()
        perturbations = perturbations.reshape(features.shape)
        directions = torch.sign(gradients) * (perturbations > 1e-6)

        if self.is_attacker:
            # 5. tailor the interdependent apis (note: application-specific)
            perturbations += (torch.sum(directions, dim=-1, keepdim=True) < 0) * checking_nonexist_api
            directions += perturbations * self.omega
        return perturbations, directions

    def get_loss(self, model, adv_x, label):
        logits_f = model.forward_f(adv_x)
        ce = F.cross_entropy(logits_f, label, reduction='none')
        y_pred = logits_f.argmax(1)
        if 'forward_g' in type(model).__dict__.keys() and (not self.oblivion):
            logits_g = model.forward_g(adv_x)
            if self.is_attacker:
                loss_no_reduction = ce + self.lambda_ * (torch.clamp(
                        model.tau - logits_g, max=self.kappa))
            else:
                loss_no_reduction = ce + self.lambda_ * (model.tau - logits_g)
            done = (y_pred == 0.) & (logits_g <= model.tau)
        else:
            loss_no_reduction = ce
            done = y_pred == 0.
        return loss_no_reduction, done
