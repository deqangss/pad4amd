from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import numpy as np

from core.defense import Dataset
from core.defense import DNNMalwareDetector, KernelDensityEstimation, AdvMalwareDetectorICNN, MaxAdvTraining, PrincipledAdvDet
from core.attack import StepwiseMax
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.stepwise_max_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for step-wise max attack')
atta_argparse.add_argument('--steps', type=int, default=50,
                           help='maximum number of steps.')
atta_argparse.add_argument('--step_length_l1', type=float, default=1.,
                           help='step length in each step of pgd l1.')
atta_argparse.add_argument('--step_length_l2', type=float, default=0.5,
                           help='step length in each step of pgd l2.')
atta_argparse.add_argument('--step_length_linf', type=float, default=0.005,
                           help='step length in each step of pgd linf.')
atta_argparse.add_argument('--random_start', action='store_true', default=False,
                           help='randomly initialize the start points.')
atta_argparse.add_argument('--round_threshold', type=float, default=0.5,
                           help='threshold for rounding real scalars at the initialization step.')
atta_argparse.add_argument('--base', type=float, default=10.,
                           help='base of a logarithm function.')
atta_argparse.add_argument('--oblivion', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--kappa', type=float, default=1.,
                           help='attack confidence.')
atta_argparse.add_argument('--real', action='store_true', default=False,
                           help='whether produce the perturbed apks.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'kde', 'amd_icnn', 'md_at_ma', 'amd_at_ma'],
                           help="model type, either of 'md_dnn', 'kde', 'amd_icnn', 'md_at_ma', and 'amd_at_ma'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx',
                           help='model timestamp.')


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
    elif args.model == 'amd_at_ma':
        save_dir = config.get('experiments', 'amd_at_ma') + '_' + args.model_name
    else:
        raise TypeError("Expected 'md_dnn', 'kde', 'amd_icnn', 'md_at_ma', and 'amd_at_ma'.")

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
    if mal_count <= 0:
        return
    mal_test_dataset_producer = dataset.get_input_producer(mal_test_x, mal_testy,
                                                           batch_size=hp_params['batch_size'],
                                                           name='test')
    assert dataset.n_classes == 2

    # test
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
    # initial model
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
    elif args.model == 'amd_at_ma':
        adv_model = PrincipledAdvDet(model)
        adv_model.load()
        model = adv_model.model
    else:
        model.load()
    logger.info("Load model parameters from {}.".format(model.model_save_path))

    model.predict(mal_test_dataset_producer, indicator_masking=True)
    attack = StepwiseMax(use_random=args.random_start,
                         rounding_threshold=args.round_threshold,
                         oblivion=args.oblivion,
                         kappa=args.kappa,
                         device=model.device
                         )

    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    model.eval()
    for x, y in mal_test_dataset_producer:
        x, y = utils.to_tensor(x.double(), y.long(), model.device)
        adv_x_batch = attack.perturb(model.double(), x, y,
                                     args.steps,
                                     args.step_length_l1,
                                     args.step_length_l2,
                                     args.step_length_linf,
                                     min_lambda_=1e-5,
                                     max_lambda_=1e5,
                                     verbose=True)
        y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch, y)
        y_cent_list.append(y_cent_batch)
        x_density_list.append(x_density_batch)
        x_mod_integrated.append((adv_x_batch - x).detach().cpu().numpy())
    y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)
    logger.info(f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.concatenate(x_density_list))
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

    save_dir = os.path.join(config.get('experiments', 'stepwise_max'), args.model)
    if not os.path.exists(save_dir):
        utils.mkdir(save_dir)
    x_mod_integrated = np.concatenate(x_mod_integrated, axis=0)
    utils.dump_pickle_frd_space(x_mod_integrated,
                                os.path.join(save_dir, 'x_mod.list'))

    if args.real:
        attack.produce_adv_mal(x_mod_integrated, mal_test_x.tolist(),
                               config.get('dataset', 'malware_dir'))


if __name__ == '__main__':
    _main()