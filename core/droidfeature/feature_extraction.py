import os.path
import warnings
from tqdm import tqdm
import multiprocessing

import collections
import numpy as np
import networkx as nx
import itertools

from core.droidfeature import sequence_generator as seq_gen
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.droidfeature.feature_extraction')
logger.addHandler(ErrorHandler)
NULL_ID = 'null'


class Apk2graphs(object):
    """Construct api graphs using api sequences that are based on the function call graphs"""

    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir,
                 number_of_sequences=200000,
                 depth_of_recursion=50,
                 timeout=6,
                 max_vocab_size=5000,
                 use_feature_selection=True,
                 use_graph_merging=True,
                 minimum_graphs_of_leaf=16,
                 maximum_graphs_of_leaf=32,
                 file_ext='.gpickle',
                 update=False,
                 proc_number=2,
                 **kwargs
                 ):
        """
        initialization
        :param naive_data_save_dir: a directory for saving intermediates
        :param intermediate_save_dir: a directory for saving meta information
        :param max_vocab_size: the maximum number of words
        :param number_of_sequences: the maximum number on the returned api sequences
        :param depth_of_recursion: the maximum depth when conducting depth-first traverse
        :param timeout: the elapsed time on analysis an app
        :param use_feature_selection: use feature selection to filtering out entities with high frequencies
        :param use_graph_merging: boolean, merge graphs or not
        :param minimum_graphs_of_leaf: integer, the minimum graphs in a node if merging graphs,
        :param maximum_graphs_of_leaf: integer, the maximum graphs in a node
        :param file_ext: file extension
        :param update: boolean indicator for recomputing the naive features
        :param proc_number: process number
        """
        self.naive_data_save_dir = naive_data_save_dir
        self.intermediate_save_dir = intermediate_save_dir
        self.use_feature_selection = use_feature_selection
        self.use_graph_merge = use_graph_merging
        self.maximum_vocab_size = max_vocab_size
        self.number_of_sequences = number_of_sequences
        self.depth_of_recursion = depth_of_recursion
        self.minimum_graphs_of_leaf = minimum_graphs_of_leaf
        self.maximum_graphs_of_leaf = maximum_graphs_of_leaf
        self.time_out = timeout

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

        params = [(apk_path, self.number_of_sequences, self.depth_of_recursion, self.time_out,
                   self.use_graph_merge, self.minimum_graphs_of_leaf, self.maximum_graphs_of_leaf,
                   get_save_path(apk_path)) for \
                 apk_path in sample_path_list if get_save_path(apk_path) is not None]
        for res in tqdm(pool.imap_unordered(seq_gen.apk2graphs_wrapper, params), total=len(params)):
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
            else:
                logger.warning("Fail to perform feature extraction for '{}'".format(apk_path))

        return feature_paths

    def get_vocab(self, feature_path_list=None, gt_labels=None):
        """
        get vocabularies incorporating feature selection
        :param feature_path_list:  feature_path_list, list, a list of paths, each of which directs to a feature file (we suggests using the feature files for the training purpose)
        :param gt_labels: gt_labels, list or numpy.ndarray, ground truth labels
        :return: list, a list of words
        """
        vocab_saving_path = os.path.join(self.intermediate_save_dir, 'data.vocab')
        vocab_extra_info_saving_path = os.path.join(self.intermediate_save_dir, 'data.vocab_info')
        if os.path.exists(vocab_saving_path) and os.path.exists(vocab_saving_path) and (not self.update):
            return utils.read_pickle(vocab_saving_path), utils.read_pickle(vocab_extra_info_saving_path)
        elif feature_path_list is None and gt_labels is None:
            raise FileNotFoundError("No vocabulary found!")
        else:
            pass
        assert len(feature_path_list) == len(gt_labels)

        counter_mal, counter_ben = collections.Counter(), collections.Counter()
        api_info_dict = collections.defaultdict(set)
        for feature_path, label in zip(feature_path_list, gt_labels):
            if not os.path.exists(feature_path):
                continue
            cg_dict = seq_gen.read_from_disk(
                feature_path)  # each file contains a dict of {root call method: networkx objects}
            api_occurence = set()
            for root_call, sub_cg in cg_dict.items():
                node_names = sub_cg.nodes(data=True)
                api_names = [api_name for api_name, _ in node_names]
                api_info = [[seq_gen.get_api_info(tag) for tag in api_tag['tag']] for _, api_tag in node_names]
                api_occurence.update(api_names)

                for an, apii_list in zip(api_names, api_info):
                    for apii in apii_list:
                        api_info_dict[an].add(apii)
            if label:
                counter_mal.update(list(api_occurence))
            else:
                counter_ben.update(list(api_occurence))
        all_words = list(set(list(counter_ben.keys()) + list(counter_mal.keys())))
        if not self.use_feature_selection:  # no feature selection applied
            return all_words
        mal_feature_frequency = np.array(list(map(counter_mal.get, all_words)))
        mal_feature_frequency[mal_feature_frequency == None] = 0
        mal_feature_frequency /= float(np.sum(gt_labels))
        ben_feature_frequency = np.array(list(map(counter_ben.get, all_words)))
        ben_feature_frequency[ben_feature_frequency == None] = 0
        ben_feature_frequency /= float(len(gt_labels) - np.sum(gt_labels))
        feature_freq_diff = abs(mal_feature_frequency - ben_feature_frequency)
        pos_selected = np.argsort(feature_freq_diff)[::-1][:self.maximum_vocab_size - 1]
        selected_words = [all_words[p] for p in pos_selected]
        corresponding_word_info = list(map(api_info_dict.get, selected_words))
        selected_words.append(NULL_ID)
        corresponding_word_info.append({NULL_ID})
        # saving
        if len(selected_words) > 0:
            utils.dump_pickle(selected_words, vocab_saving_path)
            utils.dump_pickle(corresponding_word_info, vocab_extra_info_saving_path)
        return selected_words, corresponding_word_info

    def feature_selection(self, train_features, train_y, vocab, dim):
        """
        feature selection
        :param train_features: 2D feature
        :type train_features: numpy object
        :param train_y: ground truth labels
        :param vocab: a list of words (i.e., features)
        :param dim: the number of remained words
        :return: chose vocab
        """
        is_malware = (train_y == 1)
        mal_features = np.array(train_features, dtype=object)[is_malware]
        ben_features = np.array(train_features, dtype=object)[~is_malware]

        if (len(mal_features) <= 0) or (len(ben_features) <= 0):
            return vocab

        mal_representations = self.get_feature_representation(mal_features, vocab)
        mal_frequency = np.sum(mal_representations, axis=0) / float(len(mal_features))
        ben_representations = self.get_feature_representation(ben_features, vocab)
        ben_frequency = np.sum(ben_representations, axis=0) / float(len(ben_features))

        # eliminate the words showing zero occurrence in apk files
        is_null_feature = np.all(mal_representations == 0, axis=0) & np.all(ben_representations, axis=0)
        mal_representations, ben_representations = None, None
        vocab_filtered = list(np.array(vocab)[~is_null_feature])

        if len(vocab_filtered) <= dim:
            return vocab_filtered
        else:
            feature_frq_diff = np.abs(mal_frequency[~is_null_feature] - ben_frequency[~is_null_feature])
            position_flag = np.argsort(feature_frq_diff)[::-1][:dim]

            vocab_selected = []
            for p in position_flag:
                vocab_selected.append(vocab_filtered[p])
            return vocab_selected

    def feature_mapping(self, feature_path_list, dictionary):
        """
        mapping feature to numerical representation
        :param feature_path_list: a list of feature paths
        :param dictionary: vocabulary -> index
        :return: 2D representation
        :rtype numpy.ndarray
        """
        raise NotImplementedError

    def feature2ipt(self, feature_path_list, gt_labels=None, is_adj=False, n_cg=1000):
        """
        Mapping features to the numerical representation
        :param feature_path_list, list, a list of paths, each of which directs to a feature file
        :param gt_labels, list or numpy.ndarray, ground truth labels
        :param is_adj, boolean, whether extract structural information or not
        :param n_cg, integer, the limited number of call graphs
        """
        assert len(feature_path_list) == len(gt_labels), 'inconsistent data size {} vs. label size {}'.format(
            len(feature_path_list), len(gt_labels)
        )
        if len(feature_path_list) == 0:
            return [], [], []

        features, adj, labels = [], [], []
        vocab, _ = self.get_vocab()
        representation_container = self.graph2representation(feature_path_list, gt_labels, vocab, is_adj, n_cg)
        for rpst in representation_container:
            rpst_dict, label, feature_path = rpst
            sub_features = []
            sub_adjs = []
            for root_call, sub_rpst in rpst_dict.items():
                sub_feature, sub_adj = sub_rpst
                sub_features.append(sub_feature)
                sub_adjs.append(sub_adj)
            features.append(sub_features)  # [(number_of_files, number_of_subgraphs), number_of_words]
            adj.append(sub_adjs)   # [(number of files, number_of_subgraphs), number of words, number of words]
            labels.append(label)  # [number of files,]
        return features, adj, labels

    @staticmethod
    def graph2representation(feature_path_list, gt_labels, vocabulary=None, is_adj=False, n_cg=1000):
        """
        map graphs to numerical representations :param feature_path_list, list, a list of paths, each of which
        directs to a feature file :param gt_labels, list or numpy.ndarray, ground truth labels :param vocabulary:
        list, a list of words :return: a list of numerical representations corresponds to apps. Each representation
        contains a tuple ({'root call 1': (feature, adjacent matrix), 'root call 2': (feature, adjacent matrix),
        ...}, label)
        """
        assert len(feature_path_list) == len(gt_labels)
        assert len(vocabulary) > 0

        numerical_representation_container = []
        for feature_path, label in zip(feature_path_list, gt_labels):
            if not os.path.exists(feature_path):
                logger.warning("Cannot find the feature path: {}".format(feature_path))
                continue
            cg_dict = seq_gen.read_from_disk(feature_path)
            numerical_representation_dict = collections.defaultdict(tuple)
            for i, (root_call, cg) in enumerate(cg_dict.items()):
                numerical_representation_dict[root_call] = _graph2rpst_wrapper((cg, vocabulary, is_adj))
                if len(numerical_representation_dict) > 0:
                    numerical_representation_container.append([numerical_representation_dict, label, feature_path])
                if i >= n_cg:
                    return numerical_representation_container

            # numerical_representation_dict = collections.defaultdict(tuple)
            # cpu_count = multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() // 2 > 1 else 1
            # pool = multiprocessing.Pool(cpu_count, initializer=pool_initializer)
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
        return numerical_representation_container


def _graph2rpst_wrapper(args):
    try:
        return graph2rpst(*args)
    except Exception as e:
        return e


def graph2rpst(g, vocab, is_adj):
    new_g = g.copy()
    indices = []
    from scipy.sparse import csr_matrix
    for node in g.nodes():
        if node not in vocab:
            if is_adj:
                # make connection between the predecessors and successors
                if new_g.out_degree(node) > 0 and new_g.in_degree(node) > 0:
                    new_g.add_edges_from([(e1, e1) for e1, e2 in \
                                          itertools.product(new_g.predecessors(node),
                                                            new_g.successors(node))])
                    # print([node for node in new_cg.predecessors(node)])
                    # print([node for node in cg.successors(node)])
                new_g.remove_node(node)
        else:
            indices.append(vocab.index(node))
    indices.append(vocab.index(NULL_ID))
    feature = np.zeros((len(vocab), ), dtype=np.float32)
    feature[indices] = 1.
    if is_adj:
        rear = csr_matrix(([1], ([len(vocab) - 1], [len(vocab) - 1])), shape=(len(vocab), len(vocab)))
        adj = nx.convert_matrix.to_scipy_sparse_matrix(g, nodelist=vocab, format='csr', dtype=np.float32)
        adj += rear
    else:
        adj = None
    del new_g
    del g
    return feature, adj


def _main():
    from config import config
    malware_dir_name = config.get('drebin', 'malware_dir')
    meta_data_saving_dir = config.get('drebin', 'intermediate_directory')
    naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
    feature_extractor = Apk2graphs(naive_data_saving_dir,
                                   meta_data_saving_dir,
                                   update=False,
                                   proc_number=2)
    feature_extractor.feature_extraction(malware_dir_name)


if __name__ == "__main__":
    import sys

    sys.exit(_main())
