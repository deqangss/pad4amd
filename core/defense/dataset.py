import os
import random
import tempfile
from queue import Queue

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import config
from core.droidfeature.feature_extraction import Apk2graphs
from tools import utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name='drebin', k=32, is_adj=False, undersampling_ratio=0., use_cache=False, seed=0, n_sgs_max=1000, feature_ext_args=None):
        """
        build dataset for ml model learning
        :param dataset_name: String, the dataset name, expected 'drebin' or 'androzoo'
        :param k: Integer, the number of subgraphs is sampled for passing through the neural networks
        :param is_adj: Boolean, whether use the actual adjacent matrix or not
        :param undersampling_ratio: Float, whether use the undersampling based on the ratio
        :param use_cache: Boolean, whether to use the cached data or not, the cached data is identified by a string format name
        :param seed: Integer, the random seed
        :param n_sgs_max: Integer, the maximum number of subgraphs
        :param feature_ext_args: Dict, arguments for feature extraction
        """
        self.dataset_name = dataset_name
        self.k = k
        self.is_adj = is_adj
        self.undersampling_ratio = undersampling_ratio
        if self.undersampling_ratio <= 0:
            self.use_undersampling = False
        else:
            self.use_undersampling = True
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_default_dtype(torch.float32)
        assert self.k < n_sgs_max
        self.n_sgs_max = n_sgs_max
        self.use_cache = use_cache
        self.feature_ext_args = feature_ext_args
        self.temp_dir_handle = tempfile.TemporaryDirectory()
        assert self.dataset_name in ['drebin', 'androzoo'], 'Expected either "drebin" or "androzoo".'
        if feature_ext_args is None:
            self.feature_extractor = Apk2graphs(config.get('metadata', 'naive_data_pool'),
                                                config.get(self.dataset_name, 'intermediate'))
        else:
            assert isinstance(feature_ext_args, dict)
            self.feature_extractor = Apk2graphs(config.get('metadata', 'naive_data_pool'),
                                                config.get(self.dataset_name, 'intermediate'),
                                                **feature_ext_args)

        data_saving_path = os.path.join(config.get(self.dataset_name, 'intermediate'), 'dataset.idx')
        if os.path.exists(data_saving_path):
            (self.train_dataset, self.validation_dataset, self.test_dataset) = utils.read_pickle(data_saving_path)

            def path_tran(data_paths):
                return np.array(
                    [os.path.join(config.get('metadata', 'naive_data_pool'), os.path.basename(name)) for name in data_paths])
            self.train_dataset = (path_tran(self.train_dataset[0]), self.train_dataset[1])
            self.validation_dataset = (path_tran(self.validation_dataset[0]), self.validation_dataset[1])
            self.test_dataset = (path_tran(self.test_dataset[0]), self.test_dataset[1])
        else:
            mal_feature_paths = self.apk_preprocess(
                config.get(self.dataset_name, 'malware_dir'))
            ben_feature_paths = self.apk_preprocess(
                config.get(self.dataset_name, 'benware_dir'))

            feature_paths = mal_feature_paths + ben_feature_paths
            gt_labels = np.zeros((len(mal_feature_paths) + len(ben_feature_paths)), dtype=np.int32)
            gt_labels[:len(mal_feature_paths)] = 1
            self.train_dataset, self.validation_dataset, self.test_dataset = self.data_split(feature_paths, gt_labels)
            utils.dump_pickle((self.train_dataset, self.validation_dataset, self.test_dataset), data_saving_path)
        if self.use_undersampling:
            self.train_dataset = self.undersampling(self.train_dataset, self.undersampling_ratio)

        _labels, counts = np.unique(self.train_dataset[1], return_counts=True)
        self.sample_weights = np.ones_like(_labels).astype(np.float32)
        _weights = float(np.max(counts)) / counts
        for i in range(_labels.shape[0]):
            self.sample_weights[_labels[i]] = _weights[i]

        vocab, _1, = self.feature_extractor.get_vocab(*self.train_dataset)
        self.vocab_size = len(vocab)
        self.n_classes = np.unique(self.train_dataset[1]).size

    def undersampling(self, dataset, ratio=3.0):
        """ only support for number of benign samples is bigger than malware ones"""
        feature_paths, labels = dataset
        num_of_mal = np.sum(labels)
        assert num_of_mal > 0, 'No malware, exit!'
        num_of_ben = len(labels) - np.sum(labels)
        ep_ratio = ratio if num_of_ben / num_of_mal > ratio else num_of_ben / num_of_mal
        if ep_ratio <= 1.:
            return dataset

        num_of_selected = num_of_mal * ep_ratio
        _flag = labels == 0
        feature_paths_selected = feature_paths[_flag][:num_of_selected]
        gt_labels_selected = labels[_flag][:num_of_selected]
        new_features = np.concatenate([feature_paths_selected, feature_paths[~_flag]], dim=0)
        new_gt_labels = np.concatenate([gt_labels_selected, labels[~_flag]], dim=0)
        print(len(new_features))
        print(len(new_gt_labels))
        np.random.seed(self.seed)
        np.random.shuffle(new_features)
        np.random.seed(self.seed)
        np.random.shuffle(new_gt_labels)
        return new_features, new_gt_labels

    def data_split(self, feature_paths, labels):
        assert len(feature_paths) == len(labels)
        train_dn, validation_dn, test_dn = None, None, None
        data_split_path = os.path.join(config.get(self.dataset_name, 'dataset_dir'), 'tr_te_va_split.name')
        if os.path.exists(data_split_path):
            train_dn, val_dn, test_dn = utils.read_pickle(data_split_path)

        if (train_dn is None) or (validation_dn is None) or (test_dn is None):
            data_names = [os.path.splitext(os.path.basename(path))[0] for path in feature_paths]
            train_dn, test_dn = train_test_split(data_names, test_size=0.2, random_state=self.seed, shuffle=True)
            train_dn, validation_dn = train_test_split(train_dn, test_size=0.25, random_state=self.seed, shuffle=True)
            utils.dump_pickle((train_dn, validation_dn, test_dn),
                              path=data_split_path)

        def query_path(data_names):
            return np.array([path for path in feature_paths if os.path.splitext(os.path.basename(path))[0] in data_names])

        def query_indicator(data_names):
            return [True if os.path.splitext(os.path.basename(path))[0] in data_names else False for path in feature_paths]

        train_data = query_path(train_dn)
        random.seed(self.seed)
        random.shuffle(train_data)
        train_y = labels[query_indicator(train_dn)]
        random.seed(self.seed)
        random.shuffle(train_y)
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
        """
        loading features for given a list of feature paths
        # results:
        # --->> mapping feature paths to numerical representations, incorporating cache
        # --->> features: 2d list [number of files, number of subgraphs], in which each element
        # has a vector with size [vocab_size]
        # --->> _labels: 1d list [number of files]
        # --->> adjs: 2d list [number of files, number of subgraphs], in which each element has
        # a scipy sparse matrix with size [vocab_size, vocab_size]
        """
        file_path = os.path.join(self.temp_dir_handle.name, name + '.pkl')
        if os.path.exists(file_path) and ('val' in name) and self.use_cache:
            features, adjs, labels_ = torch.load(file_path)
        else:
            features, adjs, labels_ = self.feature_extractor.feature2ipt(feature_paths, labels, self.is_adj)
        if (not os.path.exists(file_path)) and ('val' in name) and self.use_cache:
            torch.save((features, adjs, labels_), file_path)

        return features, adjs, labels_

    def collate_fn(self, batch):
        # 1. Because the number of sub graphs is different between apks, we here align a batch of data
        # pad the subgraphs if an app has subgraph smaller than self.k
        # 2. We change the sparse adjacent matrix to its tuple of (indices, values, shape), accommodating the
        # unsupported issue of dataloader
        features = [item[0] for item in batch]
        adjs = [item[1] for item in batch]
        labels_ = [item[2] for item in batch]

        batch_size = len(features)
        sample_indices = []
        features_sampled = []
        adjs_sampled = []

        batch_n_sg_max = np.max([len(feature) for feature in features])
        n_sg_used = batch_n_sg_max if batch_n_sg_max < self.n_sgs_max else self.n_sgs_max
        n_sg_used = n_sg_used if n_sg_used > self.k else self.k
        for i, feature in enumerate(features):
            replacement = True if len(feature) < n_sg_used else False
            indices = np.random.choice(len(feature), n_sg_used, replacement)
            features_sampled.append([feature[_i] for _i in indices])
            adjs_sampled.append([adjs[i][_i] for _i in indices])
            sample_indices.append(indices)

        features_sample_t = np.array([np.stack(list(feat), axis=0) for feat in zip(*features_sampled)])
        # A list (with size self.k) of sparse feature vector in the mini-batch level, in which each element
        # has the shape [batch_size, vocab_size]
        if self.is_adj:
            for i in range(batch_size):
                for j in range(self.k):
                    adjs_sampled[i][j] = utils.sparse_mx_to_torch_sparse_tensor(
                        utils.sp_to_symmetric_sp(adjs_sampled[i][j])
                    )

            adjs_sample_t = torch.stack([torch.stack(list(adj)) for adj in list(zip(*adjs_sampled))[:self.k]])
            # dataloader does not support sparse matrix
            adjs_sample_tuple = utils.tensor_coo_sp_to_ivs(adjs_sample_t)
        else:
            adjs_sample_tuple = None
        # A list (with size self.k) of sparse adjacent matrix in the mini-batch level, in which each element
        # has the shape [batch_size, vocab_size, vocab_size]
        sample_indices_t = np.array(sample_indices).T

        return features_sample_t, adjs_sample_tuple, labels_, sample_indices_t

    def get_input_producer(self, data, y, batch_size, name='train'):
        params = {'batch_size': batch_size,
                  'num_workers': self.feature_ext_args['proc_number'],
                  'collate_fn': self.collate_fn,
                  'shuffle': False}
        return torch.utils.data.DataLoader(DatasetTorch(data, y, self, name=name),
                                           worker_init_fn=lambda x: np.random.seed(torch.randint(0, 2^31, [1,])[0] + x),
                                           **params)

    def clean_up(self):
        self.temp_dir_handle.cleanup()


class DatasetTorch(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataX, datay, dataset_obj, name='train'):
        'Initialization'
        try:
            assert (name == 'train' or name == 'test' or name == 'val')
        except Exception as e:
            raise AssertionError("Only support selections: 'train', 'val' or 'test'.\n")
        self.dataX = dataX
        self.datay = datay
        self.dataset_obj = dataset_obj
        self.name = name

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataX)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        feature_path = self.dataX[index]
        y = self.datay[index]
        # Load data and get label
        x, adj, y = \
            self.dataset_obj.get_numerical_input([feature_path], [y], name=self.name + str(index))
        assert len(x) > 0 and len(adj) > 0, feature_path
        return x[0], adj[0], y[0]

