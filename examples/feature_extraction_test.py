from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.droidfeature import Apk2graphs
from config import config
from core.droidfeature import feature_extraction_cmd_md as cmd_md

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
                                   timeout=args.time_out,
                                   update=False,
                                   proc_number=args.proc_number)
    malware_features = feature_extractor.feature_extraction(malware_dir_name)
    print('The number of malware files: ', len(malware_features))
    benign_features = feature_extractor.feature_extraction(benware_dir_name)
    print('The number of benign files: ', len(benign_features))

if __name__ == '__main__':
    _main()
