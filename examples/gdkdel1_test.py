from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import torch
import argparse

import numpy as np

from core.defense import Dataset
from core.defense import DNNMalwareDetector, KernelDensityEstimation, AdvMalwareDetectorICNN, MaxAdvTraining, \
    PrincipledAdvTraining
from core.attack import GDKDEl1
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.gdkdel1_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for l1 norm based projected gradient descent attack')
atta_argparse.add_argument('--m_pertb', type=int, default=100,
                           help='maximum number of perturbations.')
atta_argparse.add_argument('--bandwidth', type=float, default=20.,
                           help='variance of Gaussian distribution.')
atta_argparse.add_argument('--penalty_factor', type=float, default=1000.,
                           help='penalty factor for density estimation.')
atta_argparse.add_argument('--n_center', type=int, default=1000,
                           help='number of centers.')
atta_argparse.add_argument('--base', type=float, default=10.,
                           help='base of a logarithm function.')
atta_argparse.add_argument('--oblivion', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--kappa', type=float, default=1.,
                           help='attack confidence.')
atta_argparse.add_argument('--n_sample_times', type=int, default=1,
                           help='data sampling times when waging attacks')
atta_argparse.add_argument('--real', action='store_true', default=False,
                           help='whether produce the perturbed apks.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'kde', 'amd_icnn', 'at_amd_pad', 'mad'],
                           help="model type, either of 'md_dnn', 'kde', 'amd_icnn', 'at_amd_pad', and 'padvtrain'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')


def _main():
    args = atta_argparse.parse_args()
    if args.model == 'md_dnn':
        save_dir = config.get('experiments', 'md_dnn') + '_' + args.model_name
    elif args.model == 'kde':
        save_dir = config.get('experiments', 'kde') + '_' + args.model_name
    elif args.model == 'amd_icnn':
        save_dir = config.get('experiments', 'amd_icnn') + '_' + args.model_name
    elif args.model == 'at_amd_pad':
        save_dir = config.get('experiments', 'm_adv_training') + '_' + args.model_name
    elif args.model == 'padvtrain':
        save_dir = config.get('experiments', 'p_adv_training') + '_' + args.model_name
    else:
        raise TypeError("Expected 'md_dnn', 'kde', 'amd_icnn', 'at_amd_pad', and 'padvtrain'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
    dataset = Dataset(use_cache=hp_params['cache'],
                      feature_ext_args={'proc_number': hp_params['proc_number']}
                      )
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
    if not (args.model == 'md_dnn' or args.model == 'kde'):
        if args.model == 'at_amd_pad' and hp_params['detector'] == 'none':
            pass
        else:
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
    elif args.model == 'at_amd_pad':
        adv_model = MaxAdvTraining(model)
        adv_model.load()
        model = adv_model.model
    elif args.model == 'padvtrain':
        adv_model = PrincipledAdvTraining(model)
        adv_model.load()
        model = adv_model.model
    else:
        model.load()
    logger.info("Load model parameters from {}.".format(model.model_save_path))
    model.predict(mal_test_dataset_producer, indicator_masking=True)

    ben_feature_vectors = []
    with torch.no_grad():
        c = args.n_center if args.n_center < ben_count else ben_count
        for ben_x, ben_y in ben_test_dataset_producer:
            ben_x, ben_y = utils.to_tensor(ben_x.double(), ben_y.long(), device=dv)
            ben_feature_vectors.append(ben_x)
            if len(ben_feature_vectors) * hp_params['batch_size'] >= c:
                break
        ben_feature_vectors = torch.vstack(ben_feature_vectors)[:c]

    attack = GDKDEl1(ben_feature_vectors,
                     args.bandwidth,
                     penalty_factor=args.penalty_factor,
                     oblivion=args.oblivion,
                     kappa=args.kappa,
                     device=model.device
                     )

    logger.info("\nThe maximum number of perturbations for each example is {}:".format(args.m_pertb))
    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    model.eval()
    for x, y in mal_test_dataset_producer:
        x, y = utils.to_tensor(x.double(), y.long(), model.device)
        adv_x_batch = attack.perturb(model, x, y,
                                     args.m_pertb,
                                     min_lambda_=1e-5,
                                     max_lambda_=1e5,
                                     verbose=True)
        y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch, y)
        y_cent_list.append(y_cent_batch)
        x_density_list.append(x_density_batch)
        x_mod_integrated.append((adv_x_batch - x).detach().cpu().numpy())
    y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)
    logger.info(
        f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.concatenate(x_density_list))
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

    save_dir = os.path.join(config.get('experiments', 'gdkdel1'), args.model)
    x_mod_integrated = np.concatenate(x_mod_integrated, axis=0)
    utils.dump_pickle_frd_space(x_mod_integrated,
                                os.path.join(save_dir, 'x_mod.list'))

    if args.real:
        attack.produce_adv_mal(x_mod_integrated, mal_test_x.tolist(),
                               config.get('dataset', 'malware_dir'))


if __name__ == '__main__':
    _main()
