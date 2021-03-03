from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from tqdm import tqdm
import argparse

import torch

from core.defense import Dataset
from core.defense import MalwareDetector
from core.defense import MalwareDetectorIndicator, PrincipledAdvTraining
from core.attack import OMPA
from tools import utils
from config import config

ompa_argparse = argparse.ArgumentParser(description='arguments for orthogonal matching pursuit attack')
ompa_argparse.add_argument('--lambda_', type=float, default=1., help='balance factor for waging attack.')
ompa_argparse.add_argument('--step_length', type=float, default=1., help='step length.')
ompa_argparse.add_argument('--n_pertb', type=int, default=10, help='maximum number of perturbations.')
ompa_argparse.add_argument('--n_sample_times', type=int, default=1, help='sample times for producing data.')
ompa_argparse.add_argument('--model', type=str, choices=['advmaldet',
                                                         'prip_adv'
                                                         ], help='model type, maldet or advmaldet.')
ompa_argparse.add_argument('--model_name', type=str, default='pro', help='model name.')


def _main():
    args = ompa_argparse.parse_args()
    if args.model == 'advmaldet':
        save_dir = config.get('experiments', 'malware_detector_indicator') + '_' + args.model_name
    elif args.model == 'prip_adv':
        save_dir = config.get('experiments', 'prip_adv_training') + '_' + args.model_name
    else:
        raise TypeError("Expected 'advmaldet' or 'prip_adv'.")

    hp_params = utils.read_pickle(os.path.join(save_dir, 'hparam.pkl'))
    dataset = Dataset(hp_params['dataset_name'],
                      k=hp_params['k'],
                      use_cache=False,
                      is_adj=False,
                      feature_ext_args={'proc_number': hp_params['proc_number']}
                      )
    test_data, testy = dataset.test_dataset
    test_dataset_producer = dataset.get_input_producer(test_data, testy,
                                                       batch_size=hp_params['batch_size'],
                                                       name='test')
    assert dataset.n_classes == 2

    # test
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'
    model = MalwareDetectorIndicator(vocab_size=dataset.vocab_size,
                                     n_classes=dataset.n_classes,
                                     device=dv,
                                     sample_weights=dataset.sample_weights,
                                     name=args.model_name,
                                     **hp_params
                                     )
    model = model.to(dv)
    if args.model == 'prip_adv':
        model = model.to(dv)
        PrincipledAdvTraining(model)
    model.load()
    print("Load model parameters from {}.".format(model.model_save_path))

    attack = OMPA(lambda_=args.lambda_,
                  n_perturbations=args.n_pertb,
                  device=model.device
                  )
    # test: accuracy
    acc = []
    for i in range(args.n_sample_times):
        mal_count = 0
        cor_pred_count = 0.
        for res in test_dataset_producer:
            x_batch, adj, y_batch = res
            x_batch, adj_batch, y_batch = utils.to_tensor(x_batch, adj, y_batch, model.device)
            mal_x_batch, mal_adj_batch, mal_y_batch, null_flag = PrincipledAdvTraining.get_mal_data(x_batch, adj_batch,
                                                                                                    y_batch)
            if not null_flag:
                adv_mal_x = attack.perturb(model, mal_x_batch, mal_adj_batch, mal_y_batch, args.step_length, verbose=True)
                _, logit = model.forward(adv_mal_x, mal_adj_batch)
                cor_pred_count += (logit.argmax(1) == 1.).sum().item()
                mal_count += adv_mal_x.size()[0]
        acc.append(cor_pred_count/float(mal_count))
        print(f'Sampling index {i + 1}: the accuracy of malware on adversarial malware is {acc[-1] * 100:.3f}%.')
    print(f'The mean accuracy is {sum(acc) / args.n_sample_times}.')


if __name__ == '__main__':
    _main()
