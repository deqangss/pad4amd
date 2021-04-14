from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse

from core.defense import Dataset
from tools.utils import ivs_to_tensor_coo_sp

from config import config

cmd_md = argparse.ArgumentParser(description='arguments for feature extraction')
cmd_md.add_argument('--proc_number', type=int, default=6,
                    help='The number of threads for features extraction.')
cmd_md.add_argument('--number_of_sequences', type=int, default=200000,
                    help='The maximum number of produced sequences for each app')
cmd_md.add_argument('--depth_of_recursion', type=int, default=50,
                    help='The maximum depth restricted on the depth-first traverse')
cmd_md.add_argument('--timeout', type=int, default=20,
                    help='The maximum elapsed time for analyzing an app')
cmd_md.add_argument('--use_feature_selection', action='store_true', default=True,
                    help='Whether use feature selection or not.')
cmd_md.add_argument('--N', type=int, default=1,
                    help='the maximum number of graphs for an app.')
cmd_md.add_argument('--max_vocab_size', type=int, default=10000,
                    help='The maximum number of vocabulary size')
cmd_md.add_argument('--update', action='store_true', default=False,
                    help='Whether update the existed features.')
args = cmd_md.parse_args()
args_dict = vars(args)


def main_():
    dataset = Dataset(is_adj=False, feature_ext_args=args_dict, use_cache=True)
    validation_data, valy = dataset.validation_dataset
    val_dataset_producer = dataset.get_input_producer(validation_data[:1000], valy[:1000], batch_size=16, name='train')
    import time
    for epoch in range(2):
        start_time = time.time()
        for idx, (x, adj, l, i) in enumerate(val_dataset_producer):
            if dataset.is_adj:
                adjs = ivs_to_tensor_coo_sp(adj)
                adj = adjs[5, 1].to_dense().numpy()
                assert np.all(adj.diagonal() == np.clip(np.sum(adj, axis=0), a_min=0., a_max=1))
                assert np.all(np.abs(adj-adj.T) < 1e-8)
        print('cost time:', time.time() - start_time)
    dataset.clean_up()

if __name__ == '__main__':
    main_()