from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import numpy as np

from core.defense import Dataset
from core.defense import MalwareDetector, MalwareDetectorIndicator, PrincipledAdvTraining, KernelDensityEstimation
from core.attack import OMPAP
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.omp_plus_attack_test')
logger.addHandler(ErrorHandler)

ompap_argparse = argparse.ArgumentParser(description='arguments for enhancing orthogonal matching pursuit attack')
ompap_argparse.add_argument('--n_pertb', type=int, default=100, help='maximum number of perturbations.')
ompap_argparse.add_argument('--kappa', type=float, default=10., help='attack confidence.')
ompap_argparse.add_argument('--ascending', action='store_true', default=False,
                            help='whether start the perturbations gradually.')
ompap_argparse.add_argument('--n_sample_times', type=int, default=1, help='sample times for producing data.')
ompap_argparse.add_argument('--kde', action='store_true', default=False,
                           help='incorporate kernel density estimation.')
ompap_argparse.add_argument('--model', type=str, default='prip_adv',
                            choices=['maldet', 'advmaldet', 'prip_adv'],
                            help="model type, either of 'maldet', 'advmaldet' and 'prip_adv'.")
ompap_argparse.add_argument('--model_name', type=str, default='pro', help='model name.')


def _main():
    args = ompap_argparse.parse_args()
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
                      is_adj=False,
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
    attack = OMPAP(is_attacker=True,
                   kappa=args.kappa,
                   device=model.device)

    # test: accuracy
    if args.ascending:
        interval = 20
    else:
        interval = args.n_pertb
    for m in range(interval, args.n_pertb + 1, interval):
        logger.info("\nThe maximum number of perturbations for each example is {}:".format(m))
        prist_acc = []
        adv_acc = []
        prist_acc_ = []
        adv_acc_ = []
        model.eval()
        for i in range(args.n_sample_times):
            prist_preds = []
            adv_preds = []
            prist_preds_ = []
            adv_preds_ = []
            for res in mal_test_dataset_producer:
                x_batch, adj, y_batch = res
                x_batch, adj_batch, y_batch = utils.to_tensor(x_batch, adj, y_batch, model.device)
                adv_x_batch = attack.perturb(model, x_batch, adj_batch, y_batch,
                                             m,
                                             min_lambda_=1e-5,
                                             max_lambda_=1e5,
                                             verbose=True)

                prist_preds.append(model.inference_batch_wise(x_batch, adj, y_batch, use_indicator=False))
                adv_preds.append(model.inference_batch_wise(adv_x_batch, adj, y_batch, use_indicator=False))
                prist_preds_.append(model.inference_batch_wise(x_batch, adj, y_batch, use_indicator=True))
                adv_preds_.append(model.inference_batch_wise(adv_x_batch, adj, y_batch, use_indicator=True))
            adv_acc.append(np.mean(np.concatenate(adv_preds)))
            prist_acc.append(np.mean(np.concatenate(prist_preds)))
            adv_acc_.append(np.mean(np.concatenate(adv_preds_)))
            prist_acc_.append(np.mean(np.concatenate(prist_preds_)))
            logger.info(
                f'Sampling {i + 1}: accuracy on pristine vs. adversarial malware is {prist_acc[-1] * 100:.3f}% vs. {adv_acc[-1] * 100:.3f}%.')
            logger.info(
                f'Sampling {i + 1} (W/ indicator): accuracy on pristine vs. adversarial malware is {prist_acc_[-1] * 100:.3f}% vs. {adv_acc_[-1] * 100:.3f}%.')
        logger.info(
            f'The mean accuracy on pristine vs. adversarial malware is {sum(prist_acc) / args.n_sample_times * 100:.3f}% vs. {sum(adv_acc) / args.n_sample_times * 100:.3f}%.')
        logger.info(
            f'The mean accuracy on pristine vs. adversarial malware is (W/ indicator) {sum(prist_acc_) / args.n_sample_times * 100:.3f}% vs. {sum(adv_acc_) / args.n_sample_times * 100:.3f}%.')


if __name__ == '__main__':
    _main()
