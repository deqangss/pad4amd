from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from core.defense import Dataset

cmd_md = argparse.ArgumentParser(description='arguments for feature extraction')
cmd_md.add_argument('--proc_number', type=int, default=2,
                    help='The number of threads for features extraction.')
cmd_md.add_argument('--number_of_sequences', type=int, default=200000,
                    help='The maximum number of produced sequences for each app')
cmd_md.add_argument('--depth_of_recursion', type=int, default=50,
                    help='The maximum depth restricted on the depth-first traverse')
cmd_md.add_argument('--timeout', type=int, default=20,
                    help='The maximum elapsed time for analyzing an app')
cmd_md.add_argument('--use_feature_selection', action='store_true', default=True,
                    help='Whether use feature selection or not.')
cmd_md.add_argument('--max_vocab_size', type=int, default=10000,
                    help='The maximum number of vocabulary size')
cmd_md.add_argument('--update', action='store_true', default=False,
                    help='Whether update the existed features.')
args = cmd_md.parse_args()
args_dict = vars(args)


def main_():
    dataset = Dataset('drebin', is_adj=True, feature_ext_args=args_dict)
    train_data, trainy = dataset.train_dataset
    train_dataset_producer = dataset.get_input_producer(train_data, trainy, batch_size=16, name='train')
    for _ in range(1):
        train_dataset_producer.reset_cursor()
        for idx, x, adj, l, sample_idx in train_dataset_producer.iteration():
            print(x.shape)
            if dataset.is_adj:
                print(str(adj.shape))


if __name__ == '__main__':
    main_()