from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.droidfeature.feature_extraction import Apk2graphs
from config import config

from tensorflow.compat.v1 import flags
import numpy as np

flags.DEFINE_integer('proc_number', 2,
                     'The number of threads for features extraction')
flags.DEFINE_integer('number_of_sequences', 200000,
                     'The maximum number of produced sequences for each app')
flags.DEFINE_integer('depth_of_recursion', 50,
                     'The maximum depth restricted on the depth-first traverse')
flags.DEFINE_integer('time_out', 20,
                     'The maximum elapsed time for analyzing an app')


def _main():
    malware_dir_name = config.get('drebin', 'malware_dir')
    benware_dir_name = config.get('drebin', 'benware_dir')
    meta_data_saving_dir = config.get('drebin', 'intermediate')
    naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
    feature_extractor = Apk2graphs(naive_data_saving_dir,
                                   meta_data_saving_dir,
                                   number_of_sequences=flags.FLAGS.number_of_sequences,
                                   depth_of_recursion=flags.FLAGS.depth_of_recursion,
                                   time_out=flags.FLAGS.time_out,
                                   update=False,
                                   proc_number=flags.FLAGS.proc_number)
    malware_features = feature_extractor.feature_extraction(malware_dir_name)
    print('The number of malware files: ', len(malware_features))
    benign_features = feature_extractor.feature_extraction(benware_dir_name)
    print('The number of benign files: ', len(benign_features))


if __name__ == '__main__':
    _main()
