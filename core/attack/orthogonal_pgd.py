"""
Evading Adversarial Example Detection Defenses with Orthogonal Projected Gradient Descent
Codes are adapted from https://github.com/v-wangg/OrthogonalPGD
"""

import torch
import torch.nn.functional as F

from core.attack import PGD
from tools.utils import get_x0, round_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.orthogonal_pgd')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120


class OrthogonalPGD(PGD):
    """
    Projected gradient descent (ascent).

    Parameters
    ---------
    @param norm, 'l2' or 'linf'
    @param project_detector: if True, take gradients of g onto f
    @param project_classifier: if True, take gradients of f onto g
    @param k, if not None, take gradients of g onto f at every kth step
    @param use_random, Boolean,  whether use random start point
    @param rounding_threshold, float, a threshold for rounding real scalars
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, norm, project_detector=False, project_classifier=False, k=1,
                 use_random=False, rounding_threshold=0.5,
                 is_attacker=True, manipulation_x=None, omega=None, device=None):
        super(OrthogonalPGD, self).__init__(norm, use_random, rounding_threshold, is_attacker,
                                            False, 1.0, manipulation_x, omega, device)
        self.k = k
        self.project_detector = project_detector
        self.project_classifier = project_classifier

    def _perturb(self,
                 model,
                 x,
                 label=None,
                 steps=10,
                 step_length=1.,
                 ):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of iterations
        @param step_length: float, the step length in each iteration
        @param lambda_, float, penalty factor
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x.clone().detach()
        batch_size = x.shape[0]
        label_adv = torch.ones_like(label).to(model.device).double()

        assert hasattr(model, 'forward_g'), 'Expected an adversary detector'
        model.eval()

        for t in range(steps):
            if t == 0 and self.use_random:
                adv_x = get_x0(adv_x, rounding_threshold=self.round_threshold, is_sample=True)

            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            # calculating gradient of classifier w.r.t. images
            logits_classifier = model.forward_f(var_adv_x)
            ce = torch.mean(F.cross_entropy(logits_classifier, label, reduction='none'))
            ce.backward()
            grad_classifier = var_adv_x.grad.detach().data  # we do not put it on cpu
            grad_classifier = self.trans_grads(grad_classifier, adv_x)

            var_adv_x.grad = None
            logits_detector = model.forward_g(var_adv_x)
            loss_detector = F.binary_cross_entropy_with_logits(logits_detector, label_adv)
            loss_detector.backward()
            grad_detector = var_adv_x.grad.detach().data
            grad_detector = self.trans_grads(grad_detector, adv_x)

            if self.project_detector:
                # using Orthogonal Projected Gradient Descent
                # projection of gradient of detector on gradient of classifier
                # then grad_d' = grad_d - (project grad_d onto grad_c)
                grad_detector_proj = grad_detector - torch.bmm(
                    (torch.bmm(grad_detector.view(batch_size, 1, -1), grad_classifier.view(batch_size, -1, 1))) / (
                            1e-20 + torch.bmm(grad_classifier.view(batch_size, 1, -1),
                                              grad_classifier.view(batch_size, -1, 1))).view(-1, 1, 1),
                    grad_classifier.view(batch_size, 1, -1)).view(grad_detector.shape)
            else:
                grad_detector_proj = grad_detector

            if self.project_classifier:
                # using Orthogonal Projected Gradient Descent
                # projection of gradient of detector on gradient of classifier
                # then grad_c' = grad_c - (project grad_c onto grad_d)
                grad_classifier_proj = grad_classifier - torch.bmm(
                    (torch.bmm(grad_classifier.view(batch_size, 1, -1), grad_detector.view(batch_size, -1, 1))) / (
                            1e-20 + torch.bmm(grad_detector.view(batch_size, 1, -1),
                                              grad_detector.view(batch_size, -1, 1))).view(-1, 1, 1),
                    grad_detector.view(batch_size, 1, -1)).view(grad_classifier.shape)
            else:
                grad_classifier_proj = grad_classifier

            if self.project_detector:
                logits_classifier[range(batch_size), 0] = logits_classifier[range(batch_size), 0] - 10.
                has_attack_succeeded = (logits_classifier.argmax(1) == 0.)[:, None].float()
            else:
                has_attack_succeeded = (logits_detector + 10. <= model.tau)[:, None].float()

            if self.k:
                # take gradients of g onto f every kth step
                if t % self.k == 0:
                    grad = grad_detector_proj
                else:
                    grad = grad_classifier_proj
            else:
                if self.project_detector:
                    grad = grad_classifier_proj * (
                                1. - has_attack_succeeded) + grad_detector_proj * has_attack_succeeded
                else:
                    grad = grad_classifier_proj * has_attack_succeeded + grad_detector_proj * (
                                1. - has_attack_succeeded)

            # if torch.any(torch.isnan(grad)):
            #     print(torch.mean(torch.isnan(grad)))
            #     print("ABORT")
            #     break
            if self.norm == 'linf':
                perturbation = torch.sign(grad)
            elif self.norm == 'l2':
                l2norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
                perturbation = torch.minimum(
                    torch.tensor(1., dtype=x.dtype, device=x.device),
                    grad / l2norm
                )
            else:
                raise ValueError("'l2' or 'linf' are expected.")
            adv_x = torch.clamp(adv_x + perturbation * step_length, min=0., max=1.)
        return adv_x

    def perturb(self, model, x, label=None,
                steps=10,
                step_length=1.,
                step_check=10,
                verbose=False):
        """
        enhance attack
        """
        assert steps >= 0 and step_check > 0 and step_length >= 0
        model.eval()
        mini_steps = [step_check] * (steps // step_check)
        mini_steps = mini_steps + [steps % step_check] if steps % step_check != 0 else mini_steps

        adv_x = x.detach().clone().to(torch.double)
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
                                        step_length
                                        )
            # round
            adv_x[~done] = round_x(pert_x_cont)
        with torch.no_grad():
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            if verbose:
                logger.info(f"pgd {self.norm}: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")
        return adv_x

    def trans_grads(self, gradients, adv_features):
        # 1. look for allowable position, because only '1--> -' and '0 --> +' are permitted
        #    1.1 api insertion
        pos_insertion = (adv_features <= 0.5) * 1 * (adv_features >= 0.)
        grad4insertion = (gradients >= 0) * pos_insertion * gradients
        # grad4insertion = (gradients > 0) * gradients
        #    2 api removal
        pos_removal = (adv_features > 0.5) * 1
        grad4removal = (gradients < 0) * (pos_removal & self.manipulation_x) * gradients
        return grad4removal + grad4insertion
