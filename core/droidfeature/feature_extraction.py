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
                 number_of_smali_files=200000,
                 max_vocab_size=2000,
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
            return utils.read_pickle(vocab_saving_path), utils.read_pickle(vocab_extra_info_saving_path), False
        elif feature_path_list is None and gt_labels is None:
            raise FileNotFoundError("No vocabulary found and no features for producing vocabulary!")
        else:
            pass
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
            for feature in features:
                feature_list, feature_info_list, feature_type_list = feat_gen.get_feature_list(feature)
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
        mal_feature_frequency[mal_feature_frequency is None] = 0
        mal_feature_frequency /= float(np.sum(gt_labels))
        ben_feature_frequency = np.array(list(map(counter_ben.get, all_words)))
        ben_feature_frequency[ben_feature_frequency is None] = 0
        ben_feature_frequency /= float(len(gt_labels) - np.sum(gt_labels))
        feature_freq_diff = abs(mal_feature_frequency - ben_feature_frequency)
        pos_selected = np.argsort(feature_freq_diff)[::-1][:self.maximum_vocab_size - 1]
        selected_words = [all_words[p] for p in pos_selected]
        selected_word_type = list(map(feat_type_dict.get, selected_words))
        selected_words_typized = selected_words[selected_word_type == feat_gen.PERMISSION]
        selected_words_typized += selected_words[selected_word_type == feat_gen.INTENT]
        selected_words_typized += selected_words[selected_word_type == feat_gen.SYS_API]
        corresponding_word_info = list(map(feat_info_dict.get, selected_words_typized))
        selected_words_typized.append(NULL_ID)
        corresponding_word_info.append({NULL_ID})
        # saving
        if len(selected_words) > 0:
            utils.dump_pickle(selected_words_typized, vocab_saving_path)
            utils.dump_pickle(corresponding_word_info, vocab_extra_info_saving_path)
        return selected_words_typized, corresponding_word_info, True

    def merge_cg(self, feature_path_list):
        pool = multiprocessing.Pool(self.proc_number)
        params = [(feature_path, self.N) for feature_path in feature_path_list if os.path.exists(feature_path)]
        for res in tqdm(pool.imap_unordered(_merge_cg, params), total=len(params)):
            if res:
                pass

    def update_cg(self, feature_path_list):
        """
        append api index into each node according to the vocabulary
        """
        vocab_saving_path = os.path.join(self.intermediate_save_dir, 'data.vocab')
        if os.path.exists(vocab_saving_path) and os.path.exists(vocab_saving_path) and (not self.update):
            vocab = utils.read_pickle(vocab_saving_path)
        else:
            raise FileNotFoundError("No vocabulary found!")
        # updating graph
        for idx, feature_path in enumerate(feature_path_list):
            if not os.path.exists(feature_path):
                continue
            cg_dict = feat_gen.read_from_disk(
                feature_path)  # each file contains a dict of {root call method: networkx objects}
            for root_call, sub_cg in cg_dict.items():
                node_names = sub_cg.nodes(data=True)
                for api, api_info in node_names:
                    if api in vocab:
                        api_info['vocab_ind'] = vocab.index(api)
                    else:
                        api_info['vocab_ind'] = len(vocab) - 1
            feat_gen.save_to_disk(cg_dict, feature_path)
            del cg_dict
        return

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
    def feature2ipt(feature_path, label, vocabulary=None, cache_dir=None):
        """
        Map features to numerical representations

        Parameters
        --------
        :param feature_path, string, a path directs to a feature file
        :param label, ground truth labels
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

        cache_feature_path = None
        if cache_dir is not None:
            cache_feature_path = os.path.join(cache_dir, os.path.basename(feature_path))
        if cache_feature_path is not None and os.path.exists(cache_feature_path):
            return utils.read_pickle_frd_space(cache_feature_path)

        cg_dict = feat_gen.read_from_disk(feature_path)
        sub_features = []
        sub_adjs = []
        for i, (root_call, cg) in enumerate(cg_dict.items()):
            res = _graph2rpst_wrapper((cg, vocabulary, is_adj))
            if isinstance(res, Exception):
                continue
            sub_feature, sub_adj = res
            sub_features.append(sub_feature)
            sub_adjs.append(sub_adj)
            if i >= n_cg:
                break

        if cache_feature_path is not None:
            utils.dump_pickle_frd_space((sub_features, sub_adjs, label), cache_feature_path)
        return sub_features, sub_adjs, label

        # numerical_representation_dict = collections.defaultdict(tuple)
        # cpu_count = multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() // 2 > 1 else 1
        # pool = multiprocessing.Pool(cpu_count, initializer=utils.pool_initializer)
        # pargs = [(cg, vocabulary, is_adj) for cg in cg_dict.values()]
        # for root_call, res in zip(list(cg_dict.keys()), pool.map(_graph2rpst_wrapper, pargs)):
        #     if not isinstance(res, Exception):
        #         (feature, adj) = res
        #         numerical_representation_dict[root_call] = (feature, adj)
        #     else:
        #         logger.error("Fail to process " + feature_path + ":" + str(res))
        # pool.close()
        # pool.join()
        # if len(numerical_representation_dict) > 0:
        #     numerical_representation_container.append([numerical_representation_dict, label, feature_path])


def _graph2rpst_wrapper(args):
    try:
        return graph2rpst(*args)
    except Exception as e:
        logger.error(str(e))
        return e


def graph2rpst(g, vocab, is_adj):
    new_g = g.copy()
    indices = []
    nodes = g.nodes(data=True)
    for api, api_info in nodes:
        if api not in vocab:
            if is_adj:
                # make connection between the predecessors and successors
                if new_g.out_degree(api) > 0 and new_g.in_degree(api) > 0:
                    new_g.add_edges_from([(e1, e1) for e1, e2 in itertools.product(new_g.predecessors(api),
                                                                                   new_g.successors(api))])
                    # print([node for node in new_cg.predecessors(node)])
                    # print([node for node in cg.successors(node)])
                new_g.remove_node(api)
        else:
            # indices.append(vocab.index(api))
            indices.append(api_info['vocab_ind'])
    # indices.append(vocab.index(NULL_ID))
    indices.append(-1)  # the last word is NULL
    feature = np.zeros((len(vocab),), dtype=np.float32)
    feature[indices] = 1.
    adj = None
    if is_adj:
        rear = csr_matrix(([1], ([len(vocab) - 1], [len(vocab) - 1])), shape=(len(vocab), len(vocab)))
        adj = nx.convert_matrix.to_scipy_sparse_matrix(g, nodelist=vocab, format='csr', dtype=np.float32)
        adj += rear
    del new_g
    del g
    return feature, adj


def _merge_cg(args):
    feature_path, N = args[0], args[1]
    cg_dict = feat_gen.read_from_disk(
        feature_path)  # each file contains a dict of {root call method: networkx objects}
    new_cg_dict = feat_gen.merge_graphs(cg_dict, N)
    feat_gen.save_to_disk(new_cg_dict, feature_path)
    return True


def _main():
    from config import config
    malware_dir_name = config.get('drebin', 'malware_dir')
    meta_data_saving_dir = config.get('drebin', 'intermediate_directory')
    naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
    feature_extractor = Apk2features(naive_data_saving_dir,
                                     meta_data_saving_dir,
                                     update=False,
                                     proc_number=2)
    feature_extractor.feature_extraction(malware_dir_name)


if __name__ == "__main__":
    import sys

    sys.exit(_main())
