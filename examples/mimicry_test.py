from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import torch
import numpy as np

from core.defense import Dataset
from core.defense import DNNMalwareDetector, KernelDensityEstimation, AdvMalwareDetectorICNN, MaxAdvTraining
from core.attack import Mimicry
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.mimicry')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for mimicry attack')
atta_argparse.add_argument('--trials', type=int, default=5,
                           help='number of benign samples for perturbing one malicious file.')
atta_argparse.add_argument('--n_ben', type=int, default=1000,
                           help='number of benign samples.')
atta_argparse.add_argument('--oblivion', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--real', action='store_true', default=False,
                           help='whether produce the perturbed apks.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'kde', 'amd_icnn', 'md_at_ma', 'mad'],
                           help="model type, either of 'md_dnn', 'kde', 'amd_icnn', 'md_at_ma', and 'padvtrain'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')


def _main():
    args = atta_argparse.parse_args()
    if args.model == 'md_dnn':
        save_dir = config.get('experiments', 'md_dnn') + '_' + args.model_name
    elif args.model == 'kde':
        save_dir = config.get('experiments', 'kde') + '_' + args.model_name
    elif args.model == 'amd_icnn':
        save_dir = config.get('experiments', 'amd_icnn') + '_' + args.model_name
    elif args.model == 'md_at_ma':
        save_dir = config.get('experiments', 'md_at_ma') + '_' + args.model_name
    elif args.model == 'padvtrain':
        save_dir = config.get('experiments', 'p_adv_training') + '_' + args.model_name
    else:
        raise TypeError("Expected 'md_dnn', 'kde', 'amd_icnn', 'md_at_ma', and 'padvtrain'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
    dataset = Dataset(use_cache=hp_params['cache'],
                      feature_ext_args={'proc_number': hp_params['proc_number']})
    test_x, testy = dataset.test_dataset
    mal_save_path = os.path.join(config.get('dataset', 'dataset_dir'), 'attack.idx')
    if not os.path.exists(mal_save_path):
        mal_test_x, mal_testy = test_x[testy == 1], testy[testy == 1]
        utils.dump_pickle_frd_space((mal_test_x, mal_testy), mal_save_path)
    else:
        mal_test_x, mal_testy = utils.read_pickle_frd_space(mal_save_path)
    mal_count = len(mal_testy)
    ben_test_x, ben_testy = test_x[testy == 0], testy[testy == 0]
    ben_count = len(ben_test_x)
    if mal_count <= 0 and ben_count <= 0:
        return
    mal_test_dataset_producer = dataset.get_input_producer(mal_test_x, mal_testy,
                                                           batch_size=hp_params['batch_size'],
                                                           name='test')
    ben_test_dataset_producer = dataset.get_input_producer(ben_test_x, ben_testy,
                                                           batch_size=hp_params['batch_size'],
                                                           name='test'
                                                           )
    # test
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
    model = DNNMalwareDetector(dataset.vocab_size,
                               dataset.n_classes,
                               device=dv,
                               name=args.model_name,
                               **hp_params
                               )
    if not (args.model == 'md_dnn' or args.model == 'kde' or args.model == 'md_at_ma'):
        model = AdvMalwareDetectorICNN(model,
                                       input_size=dataset.vocab_size,
                                       n_classes=dataset.n_classes,
                                       device=dv,
                                       sample_weights=dataset.sample_weights,
                                       name=args.model_name,
                                       **hp_params
                                       )
    model = model.to(dv).double()
    if args.model == 'kde':
        model = KernelDensityEstimation(model,
                                        n_centers=hp_params['n_centers'],
                                        bandwidth=hp_params['bandwidth'],
                                        n_classes=dataset.n_classes,
                                        ratio=hp_params['ratio']
                                        )
        model.load()
    elif args.model == 'md_at_ma':
        adv_model = MaxAdvTraining(model)
        adv_model.load()
        model = adv_model.model
    else:
        model.load()
    logger.info("Load model parameters from {}.".format(model.model_save_path))
    model.eval()
    model.predict(mal_test_dataset_producer, indicator_masking=True)
    ben_feature_vectors = []
    with torch.no_grad():
        c = args.n_ben if args.n_ben < ben_count else ben_count
        for ben_x, ben_y in ben_test_dataset_producer:
            ben_x, ben_y = utils.to_tensor(ben_x.double(), ben_y.long(), device=dv)
            ben_feature_vectors.append(ben_x)
            if len(ben_feature_vectors) * hp_params['batch_size'] >= c:
                break
        ben_feature_vectors = torch.vstack(ben_feature_vectors)[:c]

    attack = Mimicry(ben_feature_vectors, oblivion=args.oblivion, device=model.device)
    success_flag_list = []
    x_mod_list = []
    for x, y in mal_test_dataset_producer:
        x, y = utils.to_tensor(x.double(), y.long(), model.device)
        _flag, x_mod = attack.perturb(model,
                                      x,
                                      trials=args.trials,
                                      seed=0,
                                      is_apk=args.real,
                                      verbose=True)
        success_flag_list.append(_flag)
        logger.info(
            f"The attack effectiveness under mimicry attack is {np.sum(_flag) / float(len(_flag)) * 100}%.")
        x_mod_list.append(x_mod)
    success_flag = np.concatenate(success_flag_list)
    logger.info(f"The mean accuracy on perturbed malware is {(1. - np.sum(success_flag) / float(mal_count)) * 100}%.")

    if args.real:
        save_dir = os.path.join(config.get('experiments', 'mimicry'), args.model)
        adv_app_dir = os.path.join(save_dir, 'adv_apps')
        if not os.path.exists(save_dir):
            utils.mkdir(save_dir)

        # x_mod_list = utils.read_pickle_frd_space(os.path.join(save_dir, 'x_mod.list'))

        attack.produce_adv_mal(x_mod_list, mal_test_x.tolist(),
                               config.get('dataset', 'malware_dir'),
                               save_dir=adv_app_dir)
        adv_feature_paths = dataset.apk_preprocess(adv_app_dir, update_feature_extraction=False)
        # dataset.feature_preprocess(adv_feature_paths)
        ben_test_dataset_producer = dataset.get_input_producer(adv_feature_paths,
                                                               np.ones((len(adv_feature_paths, ))),
                                                               batch_size=hp_params['batch_size'],
                                                               name='test'
                                                               )
        model.predict(ben_test_dataset_producer, indicator_masking=True)


if __name__ == '__main__':
    _main()
