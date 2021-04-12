from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as path
import argparse
import time

import torch
import torch.nn.functional as F

from core.defense import Dataset
from core.defense import MalwareDetector
from tools.utils import save_args, get_group_args, to_tensor, dump_pickle, read_pickle


cmd_md = argparse.ArgumentParser(description='arguments for learning malware detector')

feature_argparse = cmd_md.add_argument_group(title='feature')
feature_argparse.add_argument('--proc_number', type=int, default=2,
                              help='The number of threads for features extraction.')
feature_argparse.add_argument('--number_of_sequences', type=int, default=200000,
                              help='The maximum number of produced sequences for each app')
feature_argparse.add_argument('--depth_of_recursion', type=int, default=50,
                              help='The maximum depth restricted on the depth-first traverse')
feature_argparse.add_argument('--timeout', type=int, default=20,
                              help='The maximum elapsed time for analyzing an app')
feature_argparse.add_argument('--use_feature_selection', action='store_true', default=True,
                              help='Whether use feature selection or not.')
feature_argparse.add_argument('--max_vocab_size', type=int, default=5000,
                              help='The maximum number of vocabulary size')
feature_argparse.add_argument('--update', action='store_true', default=False,
                              help='Whether update the existed features.')

detector_argparse = cmd_md.add_argument_group(title='detector')
detector_argparse.add_argument('--cuda', action='store_true', default=False, help='whether use cuda enable gpu or cpu.')
detector_argparse.add_argument('--seed', type=int, default=0, help='random seed.')
detector_argparse.add_argument('--embedding_dim', type=int, default=8, help='embedding dimension')
detector_argparse.add_argument('--hidden_units', type=lambda s: [int(u) for u in s.split(',')], default='16',
                               help='delimited list input, e.g., "32,32"',)
detector_argparse.add_argument('--penultimate_hidden_unit', type=int, default=64, help='dimension of penultimate layer')
detector_argparse.add_argument('--n_heads', type=int, default=2, help='number of headers')
detector_argparse.add_argument('--dropout', type=float, default=0.6, help='dropout rate')
detector_argparse.add_argument('--k', type=int, default=32, help='sampling size')
detector_argparse.add_argument('--use_fusion', action='store_true', help='whether use feature fusion or not')
detector_argparse.add_argument('--n_sample_times', type=int, default=5, help='times of sampling')
detector_argparse.add_argument('--alpha_', type=float, default=0.2, help='slope coefficient of leaky-relu or elu')
detector_argparse.add_argument('--sparse', action='store_true', default=True, help='GAT with sparse version or not.')
detector_argparse.add_argument('--smooth', action='store_true', default=False,
                               help='use smooth activation elu (rather than leaky-relu) in the GAT layer.')

detector_argparse.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
detector_argparse.add_argument('--epochs', type=int, default=10, help='number of epochs to train.')
detector_argparse.add_argument('--lr', type=float, default=0.005, help='initial learning rate.')
detector_argparse.add_argument('--weight_decay', type=float, default=5e-4, help='coefficient of weight decay')
detector_argparse.add_argument('--enable_gd_ckpt', action='store_true', default=False,
                               help='gradients checkpoint for saving GPU RAM')  # enable it with a caution in the training phase

dataset_argparse = cmd_md.add_argument_group(title='data_producer')
detector_argparse.add_argument('--is_adj', action='store_true', help='incorporate branches instruction information.')
detector_argparse.add_argument('--cache', action='store_true', help='use cache data or not.')
detector_argparse.add_argument('--n_cg', type=int, default=1000, help='limited number of call graphs.')

mode_argparse = cmd_md.add_argument_group(title='mode')
mode_argparse.add_argument('--mode', type=str, default='train', choices=['train', 'test'], required=False,
                           help='learn a model or test it.')
mode_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', required=False, help='suffix date of a tested model name.')


def _main():
    args = cmd_md.parse_args()
    dataset = Dataset(k=args.k,
                      is_adj=args.is_adj,
                      n_sgs_max=args.n_cg,
                      use_cache=args.cache,
                      feature_ext_args=get_group_args(args, cmd_md, 'feature')
                      )
    (train_data, trainy), (val_data, valy), (test_data, testy) = dataset.train_dataset, dataset.validation_dataset, dataset.test_dataset
    train_dataset_producer = dataset.get_input_producer(train_data, trainy, batch_size=args.batch_size, name='train')
    val_dataset_producer = dataset.get_input_producer(val_data, valy, batch_size=args.batch_size, name='val')
    test_dataset_producer = dataset.get_input_producer(test_data, testy, batch_size=args.batch_size, name='test')
    assert dataset.n_classes == 2

    # test: model training
    if not args.cuda:
        dv = 'cpu'
    else:
        dv = 'cuda'

    model_name = args.model_name if args.mode == 'test' else time.strftime("%Y%m%d-%H%M%S")
    model = MalwareDetector(dataset.vocab_size,
                            dataset.n_classes,
                            device=dv,
                            name=model_name,
                            **vars(args)
                            )

    model = model.to(dv)

    if args.mode == 'train':
        model.fit(train_dataset_producer,
                  val_dataset_producer,
                  epochs=args.epochs,
                  lr=args.lr,
                  weight_decay=args.weight_decay
                  )
        # human readable
        save_args(path.join(path.dirname(model.model_save_path), "hparam"), vars(args))
        # serialization for building the neural nets
        dump_pickle(vars(args), path.join(path.dirname(model.model_save_path), "hparam.pkl"))

    # test: accuracy
    model.load()
    model.predict(test_dataset_producer)

    dataset.clean_up()


if __name__ == '__main__':
    _main()
