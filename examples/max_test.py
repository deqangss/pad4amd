from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from functools import partial

import numpy as np

from core.defense import Dataset
from core.defense import MalwareDetectionDNN, PGDAdvTraining, MaxAdvTraining, KernelDensityEstimation, \
    AdvMalwareDetectorICNN, AMalwareDetectionPAD, AMalwareDetectionDLA, AMalwareDetectionDNNPlus
from core.attack import Max
from core.attack import PGD, PGDl1, OrthogonalPGD
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.max_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for l1 norm based projected gradient descent attack')

atta_argparse.add_argument('--steps_max', type=int, default=5,
                           help='maximum number of steps in max attack.')
atta_argparse.add_argument('--varepsilon', type=float, default=1e-9,
                           help='small value for checking convergence.')

atta_argparse.add_argument('--lmda', type=float, default=1.,
                           help='balance factor for waging attack.')
atta_argparse.add_argument('--steps_l1', type=int, default=100,
                           help='maximum number of perturbations.')

atta_argparse.add_argument('--steps_l2', type=int, default=100,
                           help='maximum number of steps.')
atta_argparse.add_argument('--step_length_l2', type=float, default=0.5,
                           help='step length in each step.')
atta_argparse.add_argument('--random_start', action='store_true', default=False,
                           help='randomly initialize the start points.')
atta_argparse.add_argument('--steps_linf', type=int, default=100,
                           help='maximum number of steps.')
atta_argparse.add_argument('--step_length_linf', type=float, default=0.01,
                           help='step length in each step.')
atta_argparse.add_argument('--round_threshold', type=float, default=0.5,
                           help='threshold for rounding real scalars.')

atta_argparse.add_argument('--orthogonal_v', action='store_true', default=False,
                           help='use the orthogonal version of pgd.')
atta_argparse.add_argument('--project_detector', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--project_classifier', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')

atta_argparse.add_argument('--base', type=float, default=10.,
                           help='base of a logarithm function.')
atta_argparse.add_argument('--oblivion', action='store_true', default=False,
                           help='whether know the adversary indicator or not.')
atta_argparse.add_argument('--kappa', type=float, default=1.,
                           help='attack confidence.')
atta_argparse.add_argument('--real', action='store_true', default=False,
                           help='whether produce the perturbed apks.')
atta_argparse.add_argument('--batch_size', type=int, default=128,
                           help='number of examples loaded in per batch.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['md_dnn', 'md_at_pgd', 'md_at_ma',
                                    'amd_kde', 'amd_icnn', 'amd_dla', 'amd_dnn_plus', 'amd_pad_ma'],
                           help="model type, either of 'md_dnn', 'md_at_pgd', 'md_at_ma', 'amd_kde', 'amd_icnn', "
                                "'amd_dla', 'amd_dnn_plus', 'amd_pad_ma'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx',
                           help='model timestamp.')


def _main():
    args = atta_argparse.parse_args()
    if args.model == 'md_dnn':
        save_dir = config.get('experiments', 'md_dnn') + '_' + args.model_name
    elif args.model == 'md_at_pgd':
        save_dir = config.get('experiments', 'md_at_pgd') + '_' + args.model_name
    elif args.model == 'md_at_ma':
        save_dir = config.get('experiments', 'md_at_ma') + '_' + args.model_name
    elif args.model == 'amd_kde':
        save_dir = config.get('experiments', 'amd_kde') + '_' + args.model_name
    elif args.model == 'amd_icnn':
        save_dir = config.get('experiments', 'amd_icnn') + '_' + args.model_name
    elif args.model == 'amd_dla':
        save_dir = config.get('experiments', 'amd_dla') + '_' + args.model_name
    elif args.model == 'amd_dnn_plus':
        save_dir = config.get('experiments', 'amd_dnn_plus') + '_' + args.model_name
    elif args.model == 'amd_pad_ma':
        save_dir = config.get('experiments', 'amd_pad_ma') + '_' + args.model_name
    else:
        raise TypeError("Expected 'md_dnn', 'md_at_pgd', 'md_at_ma', 'amd_kde', 'amd_icnn',"
                        "'amd_dla', 'amd_dnn_plus', and 'amd_pad_ma'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
    dataset = Dataset(feature_ext_args={'proc_number': hp_params['proc_number']})
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
                                                           batch_size=args.batch_size,
                                                           name='test')

    # test
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
    # initial model
    model = MalwareDetectionDNN(dataset.vocab_size,
                                dataset.n_classes,
                                device=dv,
                                name=args.model_name,
                                **hp_params
                                )
    if args.model == 'amd_icnn' or args.model == 'amd_pad_ma':
        model = AdvMalwareDetectorICNN(model,
                                       input_size=dataset.vocab_size,
                                       n_classes=dataset.n_classes,
                                       device=dv,
                                       name=args.model_name,
                                       **hp_params
                                       )
    model = model.to(dv).double()
    if args.model == 'md_at_pgd':
        at_wrapper = PGDAdvTraining(model)
        at_wrapper.load()
        model = at_wrapper.model
    elif args.model == 'md_at_ma':
        at_wrapper = MaxAdvTraining(model)
        at_wrapper.load()
        model = at_wrapper.model
    elif args.model == 'amd_kde':
        model = KernelDensityEstimation(model,
                                        n_centers=hp_params['n_centers'],
                                        bandwidth=hp_params['bandwidth'],
                                        n_classes=dataset.n_classes,
                                        ratio=hp_params['ratio']
                                        )
        model.load()
    elif args.model == 'amd_dla':
        model = AMalwareDetectionDLA(md_nn_model=None,
                                     input_size=dataset.vocab_size,
                                     n_classes=dataset.n_classes,
                                     device=dv,
                                     name=args.model_name,
                                     **hp_params
                                     )
        model = model.to(dv).double()
        model.load()
    elif args.model == 'amd_dnn_plus':
        model = AMalwareDetectionDNNPlus(md_nn_model=None,
                                         input_size=dataset.vocab_size,
                                         n_classes=dataset.n_classes,
                                         device=dv,
                                         name=args.model_name,
                                         **hp_params
                                         )
        model = model.to(dv).double()
        model.load()
    elif args.model == 'amd_pad_ma':
        adv_model = AMalwareDetectionPAD(model)
        adv_model.load()
        model = adv_model.model
    else:
        model.load()
        model = model.to(dv).double()
    logger.info("Load model parameters from {}.".format(model.model_save_path))
    model.predict(mal_test_dataset_producer, indicator_masking=False)

    if not args.orthogonal_v:
        pgdl1 = PGDl1(oblivion=args.oblivion, kappa=args.kappa, device=model.device)
        pgdl1.perturb = partial(pgdl1.perturb,
                                steps=args.steps_l1,
                                base=args.base,
                                verbose=False
                                )
    else:
        pgdl1 = OrthogonalPGD(norm='l1',
                              project_detector=args.project_detector,
                              project_classifier=args.project_classifier,
                              device=model.device)

        pgdl1.perturb = partial(pgdl1.perturb,
                                steps=args.steps_l1,
                                step_length=1.0,
                                verbose=False
                                )

    if not args.orthogonal_v:
        pgdl2 = PGD(norm='l2', use_random=args.random_start, rounding_threshold=args.round_threshold,
                    oblivion=args.oblivion, kappa=args.kappa, device=model.device)
        pgdl2.perturb = partial(pgdl2.perturb,
                                steps=args.steps_l2,
                                step_length=args.step_length_l2,
                                base=args.base,
                                verbose=False
                                )
    else:
        pgdl2 = OrthogonalPGD(norm='l2',
                              project_detector=args.project_detector,
                              project_classifier=args.project_classifier,
                              use_random=args.random_start,
                              rounding_threshold=args.round_threshold,
                              device=model.device)
        pgdl2.perturb = partial(pgdl2.perturb,
                                steps=args.steps_l2,
                                step_length=args.step_length_l2,
                                verbose=False
                                )
    if not args.orthogonal_v:
        pgdlinf = PGD(norm='linf', use_random=False,
                      oblivion=args.oblivion, kappa=args.kappa, device=model.device)
        pgdlinf.perturb = partial(pgdlinf.perturb,
                                  steps=args.steps_linf,
                                  step_length=args.step_length_linf,
                                  base=args.base,
                                  verbose=False
                                  )
    else:
        pgdlinf = OrthogonalPGD(norm='linf',
                                project_detector=args.project_detector,
                                project_classifier=args.project_classifier,
                                device=model.device)
        pgdlinf.perturb = partial(pgdlinf.perturb,
                                  steps=args.steps_linf,
                                  step_length=args.step_length_linf,
                                  verbose=False
                                  )

    attack = Max(attack_list=[pgdlinf, pgdl2, pgdl1],
                 varepsilon=1e-20,
                 oblivion=args.oblivion,
                 device=model.device
                 )

    model.eval()
    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    for x, y in mal_test_dataset_producer:
        x, y = utils.to_tensor(x, y.long(), model.device)
        adv_x_batch = attack.perturb(model, x.double(), y,
                                     steps_max=args.steps_max,
                                     min_lambda_=1e-5,
                                     max_lambda_=1e5,
                                     verbose=True)
        y_cent_batch, x_density_batch = model.inference_batch_wise(adv_x_batch)
        y_cent_list.append(y_cent_batch)
        x_density_list.append(x_density_batch)
        x_mod_integrated.append((adv_x_batch - x).detach().cpu().numpy())
    y_pred = np.argmax(np.concatenate(y_cent_list), axis=-1)
    logger.info(f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')
    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.concatenate(x_density_list), y_pred)
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

    dir_name = 'max' if not args.orthogonal_v else 'orthogonal_max'
    save_dir = os.path.join(config.get('experiments', dir_name), args.model)
    if not os.path.exists(save_dir):
        utils.mkdir(save_dir)
    x_mod_integrated = np.concatenate(x_mod_integrated, axis=0)
    utils.dump_pickle_frd_space(x_mod_integrated,
                                os.path.join(save_dir, 'x_mod.list'))

    if args.real:
        adv_app_dir = os.path.join(save_dir, 'adv_apps')
        # x_mod_integrated = utils.read_pickle_frd_space(os.path.join(save_dir, 'x_mod.list'))
        attack.produce_adv_mal(x_mod_integrated, mal_test_x.tolist(),
                               config.get('dataset', 'malware_dir'),
                               save_dir=adv_app_dir)

        adv_feature_paths = dataset.apk_preprocess(adv_app_dir, update_feature_extraction=True)
        # dataset.feature_preprocess(adv_feature_paths)
        adv_test_dataset_producer = dataset.get_input_producer(adv_feature_paths,
                                                               np.ones((len(adv_feature_paths, ))),
                                                               batch_size=hp_params['batch_size'],
                                                               name='test'
                                                               )
        model.predict(adv_test_dataset_producer, indicator_masking=False)


if __name__ == '__main__':
    _main()
