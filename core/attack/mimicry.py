import os
import tempfile

import torch

import copy
import numpy as np
import multiprocessing

from core.attack.base_attack import BaseAttack
from core.droidfeature import inverse_feature_extraction
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.mimicry')
logger.addHandler(ErrorHandler)

EXP_OVER_FLOW = 1e-30


class Mimicry(BaseAttack):
    """
    Mimicry attack: inject the graph of benign file into malicious ones

    Parameters
    ---------
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, oblivion=False, device=None):
        super(Mimicry, self).__init__(oblivion=oblivion, device=device)
        self.inversedorid = inverse_feature_extraction.InverseDroidFeature()

    def perturb(self, model, x, ben_x, trials=10, data_fn=None, seed=0, n_sample_times=1, is_apk=False, verbose=False):
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
        @param n_sample_times: Integer, sample times in the test phase
        @param is_apk: Boolean, whether produce apks
        @param verbose: Boolean, whether present attack information or not
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
            x_mod_list = []
            for _x in x:
                mal_cg = inverse_feature_extraction.seq_gen.read_from_disk(_x)
                mal_f_name = os.path.splitext(os.path.basename(_x))[0]
                x_mod = None
                # need more efficiency than this
                with tempfile.TemporaryDirectory() as tmpdirname:
                    ben_samples = np.random.choice(ben_x, (trials,), replace=False)
                    print(ben_samples)
                    _paths = []
                    _idc_modif = []
                    for ben_f in ben_samples:
                        ben_cg = inverse_feature_extraction.seq_gen.read_from_disk(ben_f)
                        new_cg, idx_modif = self.inversedorid.merge_features(mal_cg, ben_cg)
                        tmp_fname = os.path.join(tmpdirname, mal_f_name + '_' + os.path.basename(ben_f))
                        inverse_feature_extraction.seq_gen.save_to_disk(new_cg, tmp_fname)
                        _paths.append(tmp_fname)
                        _idc_modif.append(idx_modif)

                    # pargs = [(copy.deepcopy(mal_cg), _x, ben_f, tmpdirname) for ben_f in ben_samples]
                    # cpu_count = multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() // 2 > 1 else 1
                    # pool = multiprocessing.Pool(cpu_count, initializer=utils.pool_initializer)
                    # for res in pool.imap(_perturb_wrapper, pargs):  # keep in order
                    #     if not isinstance(res, Exception):
                    #         _paths.append(res[0])
                    #         _idc_modif.append(res[1])
                    #     else:
                    #         logger.error(str(res))
                    # pool.close()
                    # pool.join()

                    ben_y = np.zeros((trials,), dtype=np.int)
                    data_producer = data_fn(np.array(_paths), ben_y, batch_size=trials, name='test')
                    y_cent, x_density = [], []
                    for _ in range(n_sample_times):
                        x, a, y, _1 = next(iter(data_producer))
                        x, a, y = utils.to_tensor(x, a, y, model.device)
                        y_cent_, x_density_ = model.inference_batch_wise(x, a, y, use_indicator=True)
                        y_cent.append(y_cent_)
                        x_density.append(x_density_)
                    y_cent = np.mean(np.stack(y_cent, axis=1), axis=1)
                    y_pred = np.argmax(y_cent, axis=-1)
                    x_density = np.mean(np.stack(x_density, axis=1), axis=1)
                    if ('indicator' in type(model).__dict__.keys()) and (not self.oblivion):
                        attack_flag = (y_pred == 0) & (model.indicator(x_density, y_pred))
                    else:
                        attack_flag = (y_pred == 0)
                    ben_id_sel = np.argmax(attack_flag)
                    print(attack_flag)

                    if 'indicator' in type(model).__dict__.keys():
                        use_flag = (y_pred == 0) & (model.indicator(x_density, y_pred))
                    else:
                        use_flag = attack_flag

                    if is_apk:
                        idx_modif = _idc_modif[ben_id_sel]
                        x_mod = np.zeros((np.max(idx_modif) + 1, len(self.inversedorid.vocab)), dtype=np.float)
                        ben_x_list, _1, _2 = self.inversedorid.feature_extractor.feature2ipt(ben_samples[ben_id_sel],
                                                                                             label=0,
                                                                                             is_adj=False,
                                                                                             vocabulary=self.inversedorid.vocab,
                                                                                             n_cg=len(idx_modif),
                                                                                             cache_dir=None)
                        assert len(ben_x_list) <= len(idx_modif)
                        if len(ben_x_list) < len(idx_modif):
                            logger.warning("Inconsistent modification: Something in feature extraction may be incorrect!")
                        for idx in idx_modif:
                            x_mod[idx] += ben_x_list[idx]

                    if not use_flag[ben_id_sel]:
                        success_flag = np.append(success_flag, [False])
                        if verbose:
                            logger.info("Fail to perturb the file {}.".format(mal_f_name))
                    else:
                        success_flag = np.append(success_flag, [True])
                        if verbose:
                            logger.info("Success to perturb the file {}".format(mal_f_name))
                x_mod_list.append(x_mod)
        return success_flag, x_mod_list


def _perturb(mal_cg, mal_sample, ben_sample, dir_saving):
    mal_f_name = os.path.splitext(os.path.basename(mal_sample))[0]
    ben_cg = inverse_feature_extraction.seq_gen.read_from_disk(ben_sample)
    new_cg, idx_modif = inverse_feature_extraction.InverseDroidFeature.merge_features(mal_cg, ben_cg)
    tmp_fname = os.path.join(dir_saving, mal_f_name + '_' + os.path.basename(ben_sample))
    inverse_feature_extraction.seq_gen.save_to_disk(new_cg, tmp_fname)
    return tmp_fname, idx_modif


def _perturb_wrapper(args):
    try:
        return _perturb(*args)
    except Exception as e:
        return e