from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.droidfeature.feature_extraction import Apk2graphs
from config import config

import argparse


cmd_md = argparse.ArgumentParser(description='Argparser for feature extraction')
cmd_md.add_argument('--proc_number', type=int, default=2,
                    help='The number of threads for features extraction.')
cmd_md.add_argument('--number_of_sequences', type=int, default=200000,
                    help='The maximum number of produced sequences for each app')
cmd_md.add_argument('--depth_of_recursion', type=int, default=50,
                    help='The maximum depth restricted on the depth-first traverse')
cmd_md.add_argument('--time_out', type=int, default=1,
                    help='The maximum elapsed time for analyzing an app')

args = cmd_md.parse_args()


def _main():
    malware_dir_name = config.get('drebin', 'malware_dir')
    benware_dir_name = config.get('drebin', 'benware_dir')
    meta_data_saving_dir = config.get('drebin', 'intermediate')
    naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
    feature_extractor = Apk2graphs(naive_data_saving_dir,
                                   meta_data_saving_dir,
                                   number_of_sequences=args.number_of_sequences,
                                   depth_of_recursion=args.depth_of_recursion,
                                   time_out=args.time_out,
                                   update=False,
                                   proc_number=args.proc_number)
    malware_features = feature_extractor.feature_extraction(malware_dir_name)
    print('The number of malware files: ', len(malware_features))
    benign_features = feature_extractor.feature_extraction(benware_dir_name)
    print('The number of benign files: ', len(benign_features))

if __name__ == '__main__':
    _main()
