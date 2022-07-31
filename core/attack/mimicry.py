import torch

import numpy as np

from core.attack.base_attack import BaseAttack
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.mimicry')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-120


class Mimicry(BaseAttack):
    """
    Mimicry attack: inject the graph of benign file into malicious ones

    Parameters
    ---------
    @param ben_x: torch.FloatTensor, feature vectors with shape [number_of_benign_files, vocab_dim]
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, ben_x, oblivion=False, device=None):
        super(Mimicry, self).__init__(oblivion=oblivion, device=device)
        self.ben_x = ben_x

    def perturb(self, model, x, trials=10, seed=0, is_apk=False, verbose=False):
        """
        modify feature vectors of malicious apps

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, feature vectors with shape [batch_size, vocab_dim]
        @param trials: Integer, repetition times
        @param seed: Integer, random seed
        @param is_apk: Boolean, whether produce apks
        @param verbose: Boolean, whether present attack information or not
        """
        assert trials > 0
        if x is None or len(x) <= 0:
            return []
        if len(self.ben_x) <= 0:
            return x
        trials = trials if trials < len(self.ben_x) else len(self.ben_x)
        success_flag = np.array([])
        with torch.no_grad():
            torch.manual_seed(seed)
            x_mod_list = []
            for _x in x:
                indices = torch.randperm(len(self.ben_x))[:trials]
                trial_vectors = self.ben_x[indices]
                _x_fixed_one = ((1. - self.manipulation_x).float() * _x)[None, :]
                modified_x = torch.clamp(_x_fixed_one + trial_vectors, min=0., max=1.)
                modified_x, y = utils.to_tensor(modified_x.double(), torch.ones(trials,).long(), model.device)
                y_cent, x_density = model.inference_batch_wise(modified_x)
                y_pred = np.argmax(y_cent, axis=-1)
                if hasattr(model, 'indicator') and (not self.oblivion):
                    attack_flag = (y_pred == 0) & (model.indicator(x_density, y_pred))
                else:
                    attack_flag = (y_pred == 0)
                ben_id_sel = np.argmax(attack_flag)

                # check the attack effectiveness
                if 'indicator' in type(model).__dict__.keys():
                    use_flag = (y_pred == 0) & (model.indicator(x_density, y_pred))
                else:
                    use_flag = attack_flag

                if not use_flag[ben_id_sel]:
                    success_flag = np.append(success_flag, [False])
                else:
                    success_flag = np.append(success_flag, [True])

                x_mod = (modified_x[ben_id_sel] - _x).detach().cpu().numpy()
                x_mod_list.append(x_mod)
            if is_apk:
                return success_flag, np.vstack(x_mod_list)
            else:
                return success_flag, None
