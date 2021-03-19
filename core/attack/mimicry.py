import os
import tempfile

import torch
import torch.nn.functional as F

import numpy as np

from core.attack.base_attack import BaseAttack
from core.droidfeature import inverse_feature_extraction
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.mimicry')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class Mimicry(object):
    """
    Mimicry attack: inject the graph of benign file into malicious ones

    Parameters
    ---------
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, device=None):
        self.device = device
        self.inversedorid = inverse_feature_extraction.InverseDroidFeature()

    def perturb(self, model, x, ben_x, trials=10, data_fn=None, seed=0, n_sample_times=5, verbose=False):
        """
        inject the graph of benign file into malicious ones

        Parameters
        -----------
        @param model, a victim model
        @param x: List, a list of paths pointing to malicious graphs
        @param ben_x: List, a list of paths pointing to benign graphs
        @param trials: Integer, repetition times
        @param data_fn: Function, yield numerical data
        @param seed: Integer, random seed
        @param n_sample_times, Integer, sample times in the test phase
        @param verbose, Boolean, whether present attack information or not
        """
        assert trials > 0
        if x is None or len(x) <= 0:
            return []
        if len(ben_x) <= 0:
            return x
        trials = trials if trials < len(ben_x) else len(ben_x)
        np.random.seed(seed)
        success_flag = np.array([])
        with torch.no_grad():
            for _x in x:
                mal_cg = inverse_feature_extraction.seq_gen.read_from_disk(_x)
                file_name = os.path.splitext(os.path.basename(_x))[0]
                # need more efficient than this
                _paths = []
                with tempfile.TemporaryDirectory() as tmpdirname:
                    ben_samples = np.random.choice(ben_x, (trials,), replace=False)
                    for ben_f in ben_samples:
                        ben_cg = inverse_feature_extraction.seq_gen.read_from_disk(ben_f)
                        new_cg = self.inversedorid.merge_features(mal_cg, ben_cg)
                        tmp_fname = os.path.join(tmpdirname, file_name + '_' + os.path.basename(ben_f))
                        inverse_feature_extraction.seq_gen.save_to_disk(new_cg, tmp_fname)
                        _paths.append(tmp_fname)
                    ben_y = np.zeros((trials, ), dtype=np.int)
                    data_producer = data_fn(np.array(_paths), ben_y, batch_size=trials, name='test')
                    y_cent, x_dense = [], []
                    for _ in range(n_sample_times):
                        x, a, y = next(iter(data_producer))
                        x, a, y = utils.to_tensor(x, a, y, model.device)
                        y_cent_, x_dense_ = model.inference_batch_wise(x, a, y, use_indicator=True)
                        y_cent.append(y_cent_)
                        x_dense.append(x_dense_)
                    y_cent = np.mean(np.stack(y_cent, axis=1), axis=1)
                    y_pred = np.argmax(y_cent, axis=-1)
                    x_dense = np.mean(np.stack(x_dense, axis=1), axis=1)
                    if 'indicator' in type(model).__dict__.keys():
                        attack_success_flag = (y_pred == 0) & (model.indicator(x_dense, y_pred))
                    else:
                        attack_success_flag = (y_pred == 0)
                    if not np.any(attack_success_flag):
                        success_flag = np.append(success_flag, [False])
                        if verbose:
                            logger.info("Fail to perturb the file {}".format(file_name))
                    else:
                        success_flag = np.append(success_flag, [True])
                        if verbose:
                            logger.info("Success to perturb the file {}".format(file_name))
        return success_flag
