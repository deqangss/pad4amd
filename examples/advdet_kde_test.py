from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path

from core.defense import Dataset
from core.defense import MalwareDetector, KernelDensityEstimation
from tools.utils import save_args, dump_pickle
from config import config
from tools import utils

import argparse

kde_argparse = argparse.ArgumentParser(description='arguments for kernel density estimation')
kde_argparse.add_argument('--n_centers', type=int, default=500, help='number of distributions')
kde_argparse.add_argument('--bandwidth', type=float, default=20., help='variance of Gaussian kernel')
kde_argparse.add_argument('--ratio', type=float, default=0.90, help='the percentage of reminded validation examples')
kde_argparse.add_argument('--mode', type=str, default='train', choices=['train', 'test'], required=False, help='learn a model or test it.')
kde_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')


def _main():
    args = kde_argparse.parse_args()
    save_dir = config.get('experiments', 'malware_detector') + '_' + args.model_name
    hp_params = utils.read_pickle(path.join(save_dir, 'hparam.pkl'))

    dataset = Dataset(hp_params['dataset_name'],
                      k=hp_params['k'],
                      is_adj=hp_params['is_adj'],
                      feature_ext_args={'proc_number': hp_params['proc_number']}
                      )
    (train_data, trainy), (val_data, valy), (test_data, testy) = dataset.train_dataset, dataset.validation_dataset, dataset.test_dataset
    train_dataset_producer = dataset.get_input_producer(train_data, trainy, batch_size=hp_params['batch_size'], name='train')
    val_dataset_producer = dataset.get_input_producer(val_data, valy, batch_size=hp_params['batch_size'], name='val')
    test_dataset_producer = dataset.get_input_producer(test_data, testy, batch_size=hp_params['batch_size'], name='test')
    assert dataset.n_classes == 2

    # test: model training
    if not hp_params['cuda']:
        dv = 'cpu'
    else:
        dv = 'cuda'

    model = MalwareDetector(dataset.vocab_size,
                            dataset.n_classes,
                            device=dv,
                            name=args.model_name,
                            **hp_params
                            )
    model = model.to(dv)
    model.load()

    kde = KernelDensityEstimation(model,
                                  n_centers=args.n_centers,
                                  bandwidth=args.bandwidth,
                                  n_classes=dataset.n_classes,
                                  ratio=args.ratio
                                  )

    if args.mode == 'train':
        kde.fit(train_dataset_producer, val_dataset_producer)

        # human readable
        save_args(path.join(path.dirname(kde.model_save_path), "hparam"), {**vars(args), **hp_params})
        # serialization
        dump_pickle({**vars(args), **hp_params}, path.join(path.dirname(kde.model_save_path), "hparam.pkl"))
        kde.save_to_disk()

    kde.load()
    kde.predict(test_dataset_producer)


if __name__ == '__main__':
    _main()
