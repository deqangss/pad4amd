from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import numpy as np

from core.defense import Dataset
from core.defense import MalwareDetector, MalwareDetectorIndicator, PrincipledAdvTraining, KernelDensityEstimation
from core.attack import Mimicry
from tools import utils
from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.mimicry')
logger.addHandler(ErrorHandler)

atta_argparse = argparse.ArgumentParser(description='arguments for mimicry attack')
atta_argparse.add_argument('--trials', type=int, default=10, help='number of benign samples for perturbing one malicious file.')
atta_argparse.add_argument('--kde', action='store_true', default=False, help='attack model enhanced by kernel density estimation.')
atta_argparse.add_argument('--model', type=str, default='maldet',
                           choices=['maldet', 'advmaldet', 'padvtrain'],
                           help="model type, either of 'maldet', 'advmaldet' and 'padvtrain'.")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')


def _main():
    args = atta_argparse.parse_args()
    if args.model == 'maldet':
        save_dir = config.get('experiments', 'malware_detector') + '_' + args.model_name
    elif args.model == 'advmaldet':
        save_dir = config.get('experiments', 'malware_detector_indicator') + '_' + args.model_name
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
    ben_test_x = test_x[testy == 0]
    if mal_count <= 0:
        return

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
    if args.model == 'padvtrain':
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
    logger.info("Load model parameters from {}.".format(model.model_save_path))
    attack = Mimicry(device=model.device)

    model.eval()
    mal_test_dataset_producer = dataset.get_input_producer(mal_test_x, mal_testy, batch_size=hp_params['batch_size'],
                                                           name='test')
    # model.predict(mal_test_dataset_producer)
    hp_params['n_sample_times'] = 1
    success_flag = attack.perturb(model,
                                  mal_test_x,
                                  ben_test_x,
                                  trials=args.trials,
                                  data_fn=dataset.get_input_producer,
                                  seed=0,
                                  n_sample_times=hp_params['n_sample_times'],
                                  verbose=True)
    logger.info(f"The accuracy under mimicry attack is {1. - np.sum(success_flag) / float(mal_count)}.")


if __name__ == '__main__':
    _main()
