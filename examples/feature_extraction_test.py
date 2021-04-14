from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

from core.droidfeature import Apk2graphs
from config import config

cmd_md = argparse.ArgumentParser(description='arguments for feature extraction')
cmd_md.add_argument('--proc_number', type=int, default=2,
                    help='number of threads for features extraction.')
cmd_md.add_argument('--number_of_sequences', type=int, default=200000,
                    help='maximum number of produced sequences for each app')
cmd_md.add_argument('--depth_of_recursion', type=int, default=50,
                    help='maximum depth restricted on the depth-first traverse')
cmd_md.add_argument('--timeout', type=int, default=6,
                    help='maximum elapsed time (minutes) for analyzing an app')
cmd_md.add_argument('--use_feature_selection', action='store_true', default=True,
                    help='use feature selection or not.')
cmd_md.add_argument('--N', type=int, default=1,
                    help='the maximum number of graphs for an app.')
cmd_md.add_argument('--max_vocab_size', type=int, default=5000,
                    help='maximum number of vocabulary size')
cmd_md.add_argument('--update', action='store_true', default=False,
                    help='update the existed features.')

args = cmd_md.parse_args()


def _main():
    malware_dir_name = config.get('dataset', 'malware_dir')
    benware_dir_name = config.get('dataset', 'benware_dir')
    meta_data_saving_dir = config.get('dataset', 'intermediate')
    naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
    feature_extractor = Apk2graphs(naive_data_saving_dir,
                                   meta_data_saving_dir,
                                   number_of_sequences=args.number_of_sequences,
                                   depth_of_recursion=args.depth_of_recursion,
                                   timeout=args.timeout,
                                   use_feature_selection=args.use_feature_selection,
                                   N=args.N,
                                   update=args.update,
                                   proc_number=args.proc_number)
    malware_features = feature_extractor.feature_extraction(malware_dir_name)
    print('The number of malware files: ', len(malware_features))
    benign_features = feature_extractor.feature_extraction(benware_dir_name)
    print('The number of benign files: ', len(benign_features))



if __name__ == '__main__':
    _main()
