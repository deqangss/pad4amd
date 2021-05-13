from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from functools import partial

import torch

import numpy as np

from core.defense import Dataset
from core.defense import MalwareDetector, KernelDensityEstimation, MalwareDetectorIndicator, MaxAdvTraining, PrincipledAdvTraining
from core.attack import Max
from core.attack import GDKDEl1, PGDAdam, PGD, PGDl1
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.max_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for l1 norm based projected gradient descent attack')

atta_argparse.add_argument('--n_step_max', type=int, default=5,
                           help='maximum number of steps in max attack.')
atta_argparse.add_argument('--varepsilon', type=float, default=1e-9,
                           help='small value for checking convergence.')

atta_argparse.add_argument('--lambda_', type=float, default=1.,
                           help='balance factor for waging attack.')
atta_argparse.add_argument('--m_pertb', type=int, default=10,
                           help='maximum number of perturbations.')
atta_argparse.add_argument('--bandwidth', type=float, default=20.,
                           help='variance of Gaussian distribution.')
atta_argparse.add_argument('--n_benware', type=int, default=5000,
                           help='number of centers.')
atta_argparse.add_argument('--penalty_factor', type=float, default=1000.,
                           help='penalty factor for density estimation.')
atta_argparse.add_argument('--n_step_l2', type=int, default=100,
                           help='maximum number of steps.')
atta_argparse.add_argument('--step_length_l2', type=float, default=1.,
                           help='step length in each step.')
atta_argparse.add_argument('--n_step_linf', type=int, default=100,
                           help='maximum number of steps.')
atta_argparse.add_argument('--step_length_linf', type=float, default=0.01,
                           help='step length in each step.')
atta_argparse.add_argument('--n_step_adam', type=int, default=100,
                           help='maximum number of steps.')
atta_argparse.add_argument('--lr', type=float, default=0.1,
                           help='learning rate.')
atta_argparse.add_argument('--step_check', type=int, default=10,
                            help='number of steps when checking the effectiveness of continuous perturbations.')
atta_argparse.add_argument('--random_start', action='store_true', default=False,
                           help='randomly initialize the start points.')
atta_argparse.add_argument('--round_threshold', type=float, default=0.5,
                           help='threshold for rounding real scalars.')

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
                           choices=['maldet', 'kde', 'gmm', 'madvtrain', 'padvtrain'],
                           help="model type, either of 'maldet', 'kde', 'gmm', 'madvtrain', and 'padvtrain'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx',
                           help='model timestamp.')


def _main():
    args = atta_argparse.parse_args()
    if args.model == 'maldet':
        save_dir = config.get('experiments', 'malware_detector') + '_' + args.model_name
    elif args.model == 'kde':
        save_dir = config.get('experiments', 'kde') + '_' + args.model_name
    elif args.model == 'gmm':
        save_dir = config.get('experiments', 'gmm') + '_' + args.model_name
    elif args.model == 'madvtrain':
        save_dir = config.get('experiments', 'm_adv_training') + '_' + args.model_name
    elif args.model == 'padvtrain':
        save_dir = config.get('experiments', 'p_adv_training') + '_' + args.model_name
    else:
        raise TypeError("Expected 'maldet', 'kde', 'gmm', 'madvtrain', or 'padvtrain'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
    dataset = Dataset(k=hp_params['k'],
                      is_adj=hp_params['is_adj'],
                      n_sgs_max=hp_params['N'],
                      use_cache=hp_params['cache'],
                      feature_ext_args={'proc_number': hp_params['proc_number']}
                      )
    test_x, testy = dataset.test_dataset
    mal_save_path = os.path.join(config.get('dataset', 'dataset_dir'), 'attack.idx')
    if not os.path.exists(mal_save_path):
        mal_test_x, mal_testy = test_x[testy == 1], testy[testy == 1]
        from numpy import random
        mal_count = len(mal_testy) if len(mal_testy) < 1000 else 1000
        mal_test_x = random.choice(mal_test_x, mal_count, replace=False)
        mal_testy = mal_testy[:mal_count]
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
    if args.model == 'maldet' or args.model == 'kde':
        model = MalwareDetector(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=args.model_name,
                                **hp_params
                                )
    else:
        model = MalwareDetectorIndicator(vocab_size=dataset.vocab_size,
                                         n_classes=dataset.n_classes,
                                         device=dv,
                                         sample_weights=dataset.sample_weights,
                                         name=args.model_name,
                                         **hp_params
                                         )
    model = model.to(dv)
    if args.model == 'kde':
        model = KernelDensityEstimation(model,
                                        n_centers=hp_params['n_centers'],
                                        bandwidth=hp_params['bandwidth'],
                                        n_classes=dataset.n_classes,
                                        ratio=hp_params['ratio']
                                        )
        model.load()
    elif args.model == 'madvtrain':
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
    # model.predict(mal_test_dataset_producer)

    ben_hidden = []
    with torch.no_grad():
        c = args.n_benware if args.n_benware < ben_count else ben_count
        for ben_x, ben_a, ben_y, _1 in ben_test_dataset_producer:
            ben_x, ben_a, ben_y = utils.to_tensor(ben_x, ben_a, ben_y, device=dv)
            ben_x_hidden, _2 = model.forward(ben_x, ben_a)
            ben_hidden.append(ben_x_hidden)
            if len(ben_hidden) * hp_params['batch_size'] >= c:
                break
        ben_hidden = torch.vstack(ben_hidden)[:c]

    gdkde = GDKDEl1(ben_hidden,
                    args.bandwidth,
                    penalty_factor=args.penalty_factor,
                    oblivion=args.oblivion,
                    kappa=args.kappa,
                    device=model.device
                    )
    gdkde.perturb = partial(gdkde.perturb,
                            m=args.m_pertb,
                            base=args.base,
                            verbose=False
                            )

    pgdl1 = PGDl1(oblivion=args.oblivion, kappa=args.kappa, device=model.device)
    pgdl1.perturb = partial(pgdl1.perturb,
                            m=args.m_pertb,
                            base=args.base,
                            verbose=False
                            )

    pgdl2 = PGD(norm='l2', use_random=args.random_start, rounding_threshold=args.round_threshold,
                oblivion=args.oblivion, kappa=args.kappa, device=model.device)
    pgdl2.perturb = partial(pgdl2.perturb,
                            steps=args.n_step_l2,
                            step_length=args.step_length_l2,
                            step_check=args.step_check,
                            base=args.base,
                            verbose=False
                            )

    pgdlinf = PGD(norm='linf', use_random=False,
                  oblivion=args.oblivion, kappa=args.kappa, device=model.device)
    pgdlinf.perturb = partial(pgdlinf.perturb,
                              steps=args.n_step_linf,
                              step_length=args.step_length_linf,
                              step_check=args.step_check,
                              base=args.base,
                              verbose=False
                              )

    pgdadma = PGDAdam(use_random=args.random_start, rounding_threshold=args.round_threshold,
                      oblivion=args.oblivion, kappa=args.kappa, device=model.device)
    pgdadma.perturb = partial(pgdadma.perturb,
                              steps=args.n_step_adam,
                              lr=args.lr,
                              step_check=args.step_check,
                              base=args.base,
                              verbose=False)

    attack = Max(attack_list=[pgdl1,pgdl2,pgdadma,pgdlinf],
                 varepsilon=1e-9,
                 oblivion=args.oblivion,
                 device=model.device
                 )

    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    model.eval()
    for i in range(args.n_sample_times):
        y_cent, x_density = [], []
        x_mod = []
        for x, a, y, g_ind in mal_test_dataset_producer:
            x, a, y = utils.to_tensor(x, a, y, model.device)
            adv_x_batch = attack.perturb(model, x, a, y,
                                         steps_of_max=args.n_step_max,
                                         min_lambda_=1e-5,
                                         max_lambda_=1e5,
                                         verbose=True)
            y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch, a, y, use_indicator=True)
            y_cent.append(y_cent_batch)
            x_density.append(x_density_batch)
            x_mod.extend(dataset.get_modification(adv_x_batch, x, g_ind, True))
        y_cent_list.append(np.vstack(y_cent))
        x_density_list.append(np.concatenate(x_density))
        x_mod_integrated = dataset.modification_integ(x_mod_integrated, x_mod)
    y_cent = np.mean(np.stack(y_cent_list, axis=1), axis=1)
    y_pred = np.argmax(y_cent, axis=-1)
    logger.info(f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.mean(np.stack(x_density_list, axis=1), axis=1), y_pred)
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

    save_dir = os.path.join(config.get('experiments', 'max'), args.model)
    if not os.path.exists(save_dir):
        utils.mkdir(save_dir)
    utils.dump_pickle_frd_space(x_mod_integrated,
                                os.path.join(save_dir, 'x_mod.list'))

    if args.real:
        attack.produce_adv_mal(x_mod_integrated, mal_test_x.tolist(),
                               config.get('dataset', 'malware_dir'),
                               adj_mod=None)


if __name__ == '__main__':
    _main()
