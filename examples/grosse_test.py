from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import numpy as np

from core.defense import Dataset
from core.defense import MalwareDetector, KernelDensityEstimation, MalwareDetectorIndicator, MaxAdvTraining, PrincipledAdvTraining
from core.attack import Groose
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.grosse_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for grosse attack')
atta_argparse.add_argument('--lambda_', type=float, default=0.01, help='balance factor for waging attack.')
atta_argparse.add_argument('--m_pertb', type=int, default=100, help='maximum number of perturbations.')
atta_argparse.add_argument('--base', type=float, default=10., help='base of a logarithm function.')
atta_argparse.add_argument('--kappa', type=float, default=1., help='attack confidence.')
atta_argparse.add_argument('--real', action='store_true', default=False, help='whether produce the perturbed apks.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['maldet', 'kde', 'advmaldet', 'madvtrain', 'padvtrain'],
                           help="model type, either of 'maldet', 'advmaldet' and 'padvtrain'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')


def _main():
    args = atta_argparse.parse_args()
    if args.model == 'maldet':
        save_dir = config.get('experiments', 'malware_detector') + '_' + args.model_name
    elif args.model == 'kde':
        save_dir = config.get('experiments', 'kde') + '_' + args.model_name
    elif args.model == 'advmaldet':
        save_dir = config.get('experiments', 'malware_detector_indicator') + '_' + args.model_name
    elif args.model == 'madvtrain':
        save_dir = config.get('experiments', 'm_adv_training') + '_' + args.model_name
    elif args.model == 'padvtrain':
        save_dir = config.get('experiments', 'p_adv_training') + '_' + args.model_name
    else:
        raise TypeError("Expected 'maldet', 'advmaldet' or 'padvtrain'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
    dataset = Dataset(hp_params['dataset_name'],
                      k=hp_params['k'],
                      is_adj=hp_params['is_adj'],
                      feature_ext_args={'proc_number': hp_params['proc_number']}
                      )
    test_x, testy = dataset.test_dataset
    mal_test_x, mal_testy = test_x[testy == 1], testy[testy == 1]
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
    if args.model == 'maldet' or args.model == 'kde' or args.model == 'madvtrain':
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
    if args.model == 'madvtrain':
        MaxAdvTraining(model)
    if args.model == 'padvtrain':
        PrincipledAdvTraining(model)

    model.load()
    logger.info("Load model parameters from {}.".format(model.model_save_path))

    model.predict(mal_test_dataset_producer)

    attack = Groose(kappa=args.kappa, device=model.device)


    logger.info("\nThe maximum number of perturbations for each example is {}:".format(args.m_pertb))
    y_cent_list, x_density_list = [], []
    x_mod_integrated = []
    model.eval()
    for i in range(hp_params['n_sample_times']):
        y_cent, x_density = [], []
        x_mod = []
        for x, a, y, g_ind in mal_test_dataset_producer:
            x, a, y = utils.to_tensor(x, a, y, model.device)
            adv_x_batch = attack.perturb(model, x, a, y,
                                         args.m_pertb,
                                         min_lambda_=1e-5,
                                         max_lambda_=1e5,
                                         base=args.base,
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
    logger.info(
        f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

    if 'indicator' in type(model).__dict__.keys():
        indicator_flag = model.indicator(np.mean(np.stack(x_density_list, axis=1), axis=1), y_pred)
        logger.info(f"The effectiveness of indicator is {sum(~indicator_flag) / mal_count * 100:.3f}%")
        acc_w_indicator = (sum(~indicator_flag) + sum((y_pred == 1.) & indicator_flag)) / mal_count * 100
        logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {acc_w_indicator:.3f}%.')

    if args.real:
        attack.produce_adv_mal(x_mod_integrated, mal_test_x.tolist(),
                               config.get('dataset', 'malware_dir'),
                               adj_mod=None)


if __name__ == '__main__':
    _main()
