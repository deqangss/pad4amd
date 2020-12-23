import os
import random
import time
import tempfile

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import config
from core.droidfeature.feature_extraction import Apk2graphs
from tools import utils


class Dataset(object):
    def __init__(self, dataset_name='drebin', k=100, is_adj=False, use_cache=False, process_number=2, seed=0):
        """
        build dataset for ml model learning
        :param dataset_name: String, the dataset name, expected 'drebin' or 'androzoo'
        :param k: Integer, the number of subgraphs is sampled for passing through the neural networks
        :param is_adj: Boolean, whether use the actual adjacent matrix or not
        :param use_cache: Boolean, whether to use the cached data or not, the cached data is identified by a string format name
        :param process_number: Integer, the number of threads for parallel running
        :param seed: Integer, the random seed
        """
        self.dataset_name = dataset_name
        self.k = k
        self.is_adj = is_adj
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.use_cache = use_cache
        self.process_number = process_number
        self.temp_dir_handle = tempfile.TemporaryDirectory()
        assert self.dataset_name in ['drebin', 'androzoo'], 'Expected either "drebin" or "androzoo".'
        self.feature_extractor = Apk2graphs(config.get('metadata', 'naive_data_pool'),
                                            config.get(self.dataset_name, 'intermediate'),
                                            proc_number=self.process_number)
        mal_feature_paths = self.apk_preprocess(
            config.get(self.dataset_name, 'malware_dir'))
        ben_feature_paths = self.apk_preprocess(
            config.get(self.dataset_name, 'benware_dir'))

        feature_paths = mal_feature_paths + ben_feature_paths
        gt_labels = np.zeros((len(mal_feature_paths) + len(ben_feature_paths)), dtype=np.int32)
        gt_labels[:len(mal_feature_paths)] = 1

        train_dn, val_dn, test_dn = None, None, None
        data_split_path = os.path.join(config.get(self.dataset_name, 'dataset_dir'), 'tr_te_va_split.name')
        if os.path.exists(data_split_path):
            train_dn, val_dn, test_dn = utils.read_pickle(data_split_path)
        self.train_dataset, self.validation_dataset, self.test_dataset = \
            self.data_split(feature_paths, gt_labels, train_dn, val_dn, test_dn)

        vocab, _1, = self.feature_extractor.get_vocab(*self.train_dataset)
        self.vocab_size = len(vocab)
        self.n_classes = np.unique(self.train_dataset[1]).size

    def data_split(self, feature_paths, labels, train_dn=None, validation_dn=None, test_dn=None):
        assert len(feature_paths) == len(labels)
        if (train_dn is None) or (validation_dn is None) or (test_dn is None):
            data_names = [os.path.basename(path) for path in feature_paths]
            train_dn, test_dn = train_test_split(data_names, test_size=0.2, random_state=self.seed, shuffle=True)
            train_dn, validation_dn = train_test_split(train_dn, test_size=0.25, random_state=self.seed, shuffle=True)
            utils.dump_pickle((train_dn, validation_dn, test_dn),
                              path=os.path.join(config.get(self.dataset_name, 'intermediate'), 'data_name.split'))

        def query_path(data_names):
            return np.array([path for path in feature_paths if os.path.basename(path) in data_names])

        def query_indicator(data_names):
            return [True if os.path.basename(path) in data_names else False for path in feature_paths]

        train_data = query_path(train_dn)
        train_y = labels[query_indicator(train_dn)]
        val_data = query_path(validation_dn)
        val_y = labels[query_indicator(validation_dn)]
        test_data = query_path(test_dn)
        test_y = labels[query_indicator(test_dn)]
        return (train_data, train_y), (val_data, val_y), (test_data, test_y)

    def apk_preprocess(self, apk_paths, labels=None):
        if labels is None:
            return self.feature_extractor.feature_extraction(apk_paths)
        else:
            assert len(apk_paths) == len(labels), \
                'uncompilable data shape {} vs. {}'.format(len(apk_paths), len(labels))
            feature_paths = self.feature_extractor.feature_extraction(apk_paths)
            labels_ = []
            for i, feature_path in enumerate(feature_paths):
                fname = os.path.splitext(os.path.basename(feature_path))[0]
                if fname in apk_paths[i]:
                    labels_.append(labels[i])
            return feature_paths, np.array(labels_)

    def get_numerical_input(self, feature_paths, labels, name):
        # --->> mapping features to numerical representations, incorporating cache
        # --->> features: 2d list [number of files, number of subgraphs], in which each element
        # has a vector with size [vocab_size]
        # --->> adjs: 2d list [number of files, number of subgraphs], in which each element has
        # a scipy sparse matrix with size [vocab_size, vocab_size]
        file_path = os.path.join(self.temp_dir_handle.name, name + '.pkl')
        if os.path.exists(file_path) and self.use_cache:
            features, adjs, labels_ = utils.read_pickle(file_path)
        else:
            features, adjs, labels_ = self.feature_extractor.feature2ipt(feature_paths, labels, self.is_adj)
        if (not os.path.exists(file_path)) and self.use_cache:
            utils.dump_pickle((features, adjs, labels_), file_path)

        # sampling subgraphs and list transpose
        batch_size = len(features)
        sample_indices = []
        features_sample = []
        adjs_sample = []
        for i, feature in enumerate(features):
            n_sg = len(feature)
            replacement = True if n_sg < self.k else False
            indices = np.random.choice(n_sg, self.k, replacement)
            features_sample.append([feature[_i] for _i in indices])
            adjs_sample.append([adjs[i][_i] for _i in indices])
            sample_indices.append(indices)
        features_sample_t = np.array([np.stack(list(feat), axis=0) for feat in zip(*features_sample)])
        # A list (with size self.k) of sparse feature vector in the mini-batch level, in which each element
        # has the shape [batch_size, vocab_size]
        if self.is_adj:
            for i in range(batch_size):
                for _k in range(self.k):
                    adjs_sample[i][_k] = utils.sparse_mx_to_torch_sparse_tensor(
                        utils.sp_to_symmetric_sp_mat(adjs_sample[i][_k])
                    )
            adjs_sample_t = torch.stack([torch.stack(list(adj), dim=0) for adj in zip(*adjs_sample)], dim=0)
        else:
            adjs_sample_t = None
        # A list (with size self.k) of sparse adjacent matrix in the mini-batch level, in which each element
        # has the shape [batch_size, vocab_size, vocab_size]
        sample_indices_t = np.array(sample_indices).T

        return features_sample_t, adjs_sample_t, labels_, sample_indices_t

    def get_input_producer(self, data, y, batch_size, name='train'):
        return _DataProducer(self, data, y, batch_size, name=name)

    def clean_up(self):
        self.temp_dir_handle.cleanup()


class _DataProducer(object):
    def __init__(self, dataset_obj, dataX, datay, batch_size, n_epochs=1, n_steps=None, name='train'):
        '''
        The data factory yield data at designated batch size and steps
        :param dataset_obj, class Dataset
        :param dataX: 2-D array numpy type supported. shape: [num, feat_dims]
        :param datay: 2-D or 1-D array.
        :param batch_size: setting batch size for training or testing. Only integer supported.
        :param n_epochs: setting epoch for training. The default value is None
        :param n_steps: setting global steps for training. The default value is None. If provided, param n_epochs will be neglected.
        :param name: 'train' or 'test'. if the value is 'test', the n_epochs will be set to 1.
        '''
        try:
            assert (name == 'train' or name == 'test' or name == 'val')
        except Exception as e:
            raise AssertionError("Only support selections: 'train', 'val' or 'test'.\n")
        self.dataset_obj = dataset_obj
        self.dataX = dataX
        self.datay = datay
        self.batch_size = batch_size
        self.mini_batches = self.dataX.shape[0] // self.batch_size
        if self.dataX.shape[0] % self.batch_size > 0:
            self.mini_batches = self.mini_batches + 1
            if (self.dataX.shape[0] > self.batch_size) and \
                    (name == 'train' or name == 'val'):
                np.random.seed(0)
                rdm_idx = np.random.choice(self.dataX.shape[0],
                                           self.batch_size - self.dataX.shape[0] % self.batch_size,
                                           replace=False)
                if len(np.array(dataX).shape) >= 2:
                    self.dataX = np.vstack([dataX, dataX[rdm_idx]])
                else:
                    self.dataX = np.concatenate([dataX, dataX[rdm_idx]])
                self.datay = np.concatenate([datay, datay[rdm_idx]])

        if name == 'train':
            if n_epochs is not None:
                self.steps = n_epochs * self.mini_batches
            elif n_steps is not None:
                self.steps = n_steps
            else:
                self.steps = None
        if name == 'test' or name == 'val':
            self.steps = None

        self.name = name
        self.cursor = 0
        if self.steps is None:
            self.max_iterations = self.mini_batches
        else:
            self.max_iterations = self.steps

    def iteration(self):
        while self.cursor < self.max_iterations:
            pos_cursor = self.cursor % self.mini_batches
            start_i = pos_cursor * self.batch_size

            end_i = start_i + self.batch_size
            if end_i > self.dataX.shape[0]:
                end_i = self.dataX.shape[0]
            if start_i == end_i:
                break
            x, adj, y, idx = self.dataset_obj.get_numerical_input(self.dataX[start_i:end_i],
                                                                  self.datay[start_i: end_i],
                                                                  name=self.name + str(self.cursor))
            yield self.cursor, x, adj, y, idx
            self.cursor = self.cursor + 1

    def reset_cursor(self):
        self.cursor = 0

    def get_current_cursor(self):
        return self.cursor
