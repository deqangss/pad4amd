import os.path
from tqdm import tqdm
import multiprocessing

import collections
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import itertools

from core.droidfeature import feature_gen as feat_gen
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.droidfeature.feature_extraction')
logger.addHandler(ErrorHandler)
NULL_ID = 'null'


class Apk2features(object):
    """Get features from an APK"""

    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir,
                 number_of_smali_files=1000000,
                 max_vocab_size=5000,
                 use_top_disc_features=True,
                 file_ext='.feat',
                 update=False,
                 proc_number=2,
                 **kwargs
                 ):
        """
        initialization
        :param naive_data_save_dir: a directory for saving intermediates
        :param intermediate_save_dir: a directory for saving feature pickle files
        :param number_of_smali_files: the maximum number of smali files processed
        :param max_vocab_size: the maximum number of words
        :param use_top_disc_features: use feature selection to filtering out entities with low discriminant
        :param file_ext: file extension
        :param update: boolean indicator for recomputing the naive features
        :param proc_number: process number
        """
        self.naive_data_save_dir = naive_data_save_dir
        self.intermediate_save_dir = intermediate_save_dir
        self.use_feature_selection = use_top_disc_features
        self.maximum_vocab_size = max_vocab_size
        self.number_of_smali_files = number_of_smali_files

        self.file_ext = file_ext
        self.update = update
        self.proc_number = proc_number

        if len(kwargs) > 0:
            logger.warning("unused hyper parameters {}.".format(kwargs))

    def feature_extraction(self, sample_dir):
        """ save the android features and return the saved paths """
        sample_path_list = utils.check_dir(sample_dir)
        pool = multiprocessing.Pool(self.proc_number, initializer=utils.pool_initializer)

        def get_save_path(a_path):
            sha256_code = os.path.splitext(os.path.basename(a_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)

            if os.path.exists(save_path) and (not self.update):
                return
            else:
                return save_path

        params = [(apk_path, self.number_of_smali_files, get_save_path(apk_path)) for \
                  apk_path in sample_path_list if get_save_path(apk_path) is not None]
        for res in tqdm(pool.imap_unordered(feat_gen.apk2feat_wrapper, params), total=len(params)):
            if isinstance(res, Exception):
                logger.error("Failed processing: {}".format(str(res)))
        pool.close()
        pool.join()

        feature_paths = []
        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)
            if os.path.exists(save_path):
                feature_paths.append(save_path)

        return feature_paths

    def get_vocab(self, feature_path_list=None, gt_labels=None):
        """
        get vocabularies incorporating feature selection
        :param feature_path_list: feature_path_list, list, a list of paths, each of which directs to a feature file (we \
        suggest using the feature files for the training purpose)
        :param gt_labels: gt_labels, list or numpy.ndarray, ground truth labels
        :return: list, a list of words
        """
        vocab_saving_path = os.path.join(self.intermediate_save_dir, 'data.vocab')
        vocab_extra_info_saving_path = os.path.join(self.intermediate_save_dir, 'data.vocab_info')
        if os.path.exists(vocab_saving_path) and os.path.exists(vocab_saving_path) and (not self.update):
            return utils.read_pickle(vocab_saving_path), utils.read_pickle(vocab_extra_info_saving_path)
        elif feature_path_list is None and gt_labels is None:
            raise FileNotFoundError("No vocabulary found and no features for producing vocabulary!")
        else:
            pass
        assert not (np.all(gt_labels == 1) or np.all(gt_labels == 0)), 'Expect both malware and benign samples.'
        assert len(feature_path_list) == len(gt_labels)

        counter_mal, counter_ben = collections.Counter(), collections.Counter()
        feat_info_dict = collections.defaultdict(set)
        feat_type_dict = collections.defaultdict(str)
        for feature_path, label in zip(feature_path_list, gt_labels):
            if not os.path.exists(feature_path):
                continue
            features = feat_gen.read_from_disk(
                feature_path)  # each file contains a dict of {root call method: networkx objects}
            feature_occurrence = set()
            feature_list, feature_info_list, feature_type_list = feat_gen.get_feature_list(features)
            feature_occurrence.update(feature_list)
            for _feat, _feat_info, _feat_type in zip(feature_list, feature_info_list, feature_type_list):
                feat_info_dict[_feat].add(_feat_info)
                feat_type_dict[_feat] = _feat_type
            if label:
                counter_mal.update(list(feature_occurrence))
            else:
                counter_ben.update(list(feature_occurrence))
        all_words = list(set(list(counter_ben.keys()) + list(counter_mal.keys())))
        if not self.use_feature_selection:  # no feature selection applied
            self.maximum_vocab_size = len(all_words) + 1
        mal_feature_frequency = np.array(list(map(counter_mal.get, all_words)))
        mal_feature_frequency[mal_feature_frequency == None] = 0
        mal_feature_frequency /= float(np.sum(gt_labels))
        ben_feature_frequency = np.array(list(map(counter_ben.get, all_words)))
        ben_feature_frequency[ben_feature_frequency == None] = 0
        ben_feature_frequency /= float(len(gt_labels) - np.sum(gt_labels))
        feature_freq_diff = abs(mal_feature_frequency - ben_feature_frequency)
        pos_selected = np.argsort(feature_freq_diff)[::-1][:self.maximum_vocab_size]
        selected_words = np.array([all_words[p] for p in pos_selected])
        selected_word_type = list(map(feat_type_dict.get, selected_words))
        print(np.array(selected_word_type) == feat_gen.PERMISSION)
        import sys
        sys.exit(1)
        selected_words_typized = (selected_words[np.array(selected_word_type) == feat_gen.PERMISSION]).tolist()
        selected_words_typized += (selected_words[np.array(selected_word_type) == feat_gen.INTENT]).tolist()
        selected_words_typized += (selected_words[np.array(selected_word_type) == feat_gen.SYS_API]).tolist()
        corresponding_word_info = list(map(feat_info_dict.get, selected_words_typized))
        # saving
        if len(selected_words) > 0:
            utils.dump_pickle(selected_words_typized, vocab_saving_path)
            utils.dump_pickle(corresponding_word_info, vocab_extra_info_saving_path)
        return selected_words_typized, corresponding_word_info

    def feature_mapping(self, feature_path_list, dictionary):
        """
        mapping feature to numerical representation
        :param feature_path_list: a list of feature paths
        :param dictionary: vocabulary -> index
        :return: 2D representation
        :rtype numpy.ndarray
        """
        raise NotImplementedError

    @staticmethod
    def get_non_api_size(vocabulary=None):
        cursor = 0
        for word in vocabulary:
            if '->' not in word:  # exclude the api features
                cursor += 1
            else:
                break
        return cursor

    @staticmethod
    def feature2ipt(feature_path, label, vocabulary=None):
        """
        Map features to numerical representations

        Parameters
        --------
        :param feature_path, string, a path directs to a feature file
        :param label, int, ground truth labels
        :param vocabulary:list, a list of words
        :return: numerical representations corresponds to an app. Each representation contains a tuple
        ([feature 1D array, api adjacent 2D array], label)
        """
        assert vocabulary is not None and len(vocabulary) > 0

        if not isinstance(feature_path, str):
            return [], [], []

        if not os.path.exists(feature_path):
            logger.warning("Cannot find the feature path: {}".format(feature_path))
            return [], [], []

        # handle the multiple modalities in vocabulary
        cursor = Apk2features.get_non_api_size(vocabulary)

        features = feat_gen.read_from_disk(feature_path)
        non_api_features, api_features = feat_gen.format_feature(features)

        # cope with the non-api features
        non_api_represenstation = np.zeros((cursor, ), dtype=np.float32)
        dictionary = dict(zip(vocabulary, range(len(vocabulary))))
        filled_pos = [idx for idx in list(map(dictionary.get, non_api_features)) if idx is not None]
        if len(filled_pos) > 0:
            non_api_represenstation[filled_pos] = 1.

        # cope with api features
        api_graph_tempate = nx.Graph()
        api_graph_tempate.add_nodes_from(vocabulary[cursor:])

        # api_representation = np.zeros(shape=(len(api_features), len(vocabulary) - cursor,
        #                                      len(vocabulary) - cursor), dtype=np.int)
        api_representations = []
        for i, api_feat in enumerate(api_features): # class wise
            api_graph_class_wise = api_graph_tempate.copy()
            for a, b in itertools.product(api_feat, api_feat):
                api_graph_class_wise.add_edge(a, b)
            api_representations.append(nx.convert_matrix.to_scipy_sparse_matrix(api_graph_class_wise))
        return non_api_represenstation, api_representations, label


def _main():
    from config import config
    malware_dir_name = config.get('dataset', 'malware_dir')
    benign_dir_name = config.get('dataset', 'benware_dir')
    meta_data_saving_dir = config.get('dataset', 'intermediate')
    naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
    feature_extractor = Apk2features(naive_data_saving_dir,
                                     meta_data_saving_dir,
                                     update=False,
                                     proc_number=2)
    mal_paths = feature_extractor.feature_extraction(malware_dir_name)
    ben_paths = feature_extractor.feature_extraction(benign_dir_name)
    labels = np.zeros((len(mal_paths) + len(ben_paths), ))
    labels[:len(mal_paths)] = 1
    vocab, _1 = feature_extractor.get_vocab(mal_paths + ben_paths, labels)
    n_rpst, api_rpst, _1 = feature_extractor.feature2ipt(mal_paths[0], label=1, vocabulary=vocab)
    print(n_rpst.shape)
    print(api_rpst)


if __name__ == "__main__":
    import sys

    sys.exit(_main())
