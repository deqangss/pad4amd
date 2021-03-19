from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from tqdm import tqdm
import argparse

import numpy as np

from core.defense import Dataset
from core.defense import MalwareDetector, MalwareDetectorIndicator, PrincipledAdvTraining, KernelDensityEstimation
from core.attack import OMPA
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.omp_attack_test')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for orthogonal matching pursuit attack')
atta_argparse.add_argument('--lambda_', type=float, default=0.01, help='balance factor for waging attack.')
atta_argparse.add_argument('--step_length', type=float, default=1., help='step length.')
atta_argparse.add_argument('--n_pertb', type=int, default=100, help='maximum number of perturbations.')
atta_argparse.add_argument('--kappa', type=float, default=10., help='attack confidence.')
atta_argparse.add_argument('--ascending', action='store_true', default=False,
                           help='whether start the perturbations gradually.')
atta_argparse.add_argument('--n_sample_times', type=int, default=1, help='sample times for producing data.')
atta_argparse.add_argument('--kde', action='store_true', default=False, help='incorporate kernel density estimation.')
atta_argparse.add_argument('--model', type=str, default='prip_adv',
                           choices=['maldet', 'advmaldet', 'prip_adv'],
                           help="model type, either of 'maldet', 'advmaldet' and 'prip_adv'.")
atta_argparse.add_argument('--model_name', type=str, default='pro', help='model name.')


def _main():
    args = atta_argparse.parse_args()
    if args.model == 'maldet':
        save_dir = config.get('experiments', 'malware_detector') + '_' + args.model_name
    elif args.model == 'advmaldet':
        save_dir = config.get('experiments', 'malware_detector_indicator') + '_' + args.model_name
    elif args.model == 'prip_adv':
        save_dir = config.get('experiments', 'prip_adv_training') + '_' + args.model_name
    else:
        raise TypeError("Expected 'maldet', 'advmaldet' or 'prip_adv'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
    dataset = Dataset(hp_params['dataset_name'],
                      k=hp_params['k'],
                      use_cache=False,
                      is_adj=hp_params['is_adj'],
                      feature_ext_args={'proc_number': hp_params['proc_number']}
                      )
    test_x, testy = dataset.test_dataset 
    mal_test_x = test_x[testy == 1]
    mal_testy = testy[testy == 1]
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
    if args.model == 'maldet':
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
    if args.model == 'prip_adv':
        PrincipledAdvTraining(model)
    if args.kde:
        save_dir = config.get('experiments', 'kde') + '_' + args.model_name
        hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
        model = KernelDensityEstimation(model,
                                        n_centers=hp_params['n_centers'],
                                        bandwidth=hp_params['bandwidth'],
                                        n_classes=dataset.n_classes,
                                        ratio=hp_params['ratio']
                                        )

    model.load()
    print("Load model parameters from {}.".format(model.model_save_path))
    logger.info(f"\n The threshold is {model.tau}.")

    model.predict(mal_test_dataset_producer, use_indicator=False)
    model.predict(mal_test_dataset_producer, use_indicator=True)

    attack = OMPA(is_attacker=True,
                  device=model.device,
                  kappa=args.kappa
                  )
    # test: accuracy
    if args.ascending:
        interval = 10
    else:
        interval = args.n_pertb

    for m in range(interval, args.n_pertb + 1, interval):
        logger.info("\nThe maximum number of perturbations for each example is {}:".format(m))
        y_cent_list, x_dense_list = [], []
        model.eval()
        for i in range(args.n_sample_times):
            y_cent, x_dense = [], []
            for res in mal_test_dataset_producer:
                x_batch, adj, y_batch = res
                x_batch, adj_batch, y_batch = utils.to_tensor(x_batch, adj, y_batch, model.device)
                adv_x_batch = attack.perturb(model, x_batch, adj_batch, y_batch,
                                             m,
                                             args.lambda_,
                                             args.step_length,
                                             verbose=False)
                y_cent_batch, x_dense_batch = model.inference_batch_wise(adv_x_batch, adj, y_batch, use_indicator=True)
                y_cent.append(y_cent_batch)
                x_dense.append(x_dense_batch)
            y_cent_list.append(np.vstack(y_cent))
            x_dense_list.append(np.concatenate(x_dense))

        y_cent = np.mean(np.stack(y_cent_list, axis=1), axis=1)
        y_pred = np.argmax(y_cent, axis=-1)
        logger.info(
            f'The mean accuracy on perturbed malware is {sum(y_pred == 1.) / mal_count * 100:.3f}%')

        if 'indicator' in type(model).__dict__.keys():
            indicator_flag = model.indicator(np.mean(np.stack(x_dense_list, axis=1), axis=1), y_pred)
            logger.info(f"The effectiveness of indicator is {np.sum(~indicator_flag) / float(len(indicator_flag)) * 100}%")
            if np.sum(~indicator_flag) < len(indicator_flag):
                c = len(indicator_flag) - np.sum(~indicator_flag)
                logger.info(f'The mean accuracy on adversarial malware (w/ indicator) is {sum((y_pred == 1.) & indicator_flag) / c * 100:.3f}%.')

if __name__ == '__main__':
    _main()
