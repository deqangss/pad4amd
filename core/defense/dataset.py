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
    def __init__(self, dataset_name='drebin', k=100, is_adj=False, use_cache=False, seed=0, max_num_sg=1024, feature_ext_args=None):
        """
        build dataset for ml model learning
        :param dataset_name: String, the dataset name, expected 'drebin' or 'androzoo'
        :param k: Integer, the number of subgraphs is sampled for passing through the neural networks
        :param is_adj: Boolean, whether use the actual adjacent matrix or not
        :param use_cache: Boolean, whether to use the cached data or not, the cached data is identified by a string format name
        :param seed: Integer, the random seed
        :param max_num_sg: Integer, the maximum number of subgraphs
        :param feature_ext_args: Dict, arguments for feature extraction
        """
        self.dataset_name = dataset_name
        self.k = k
        self.is_adj = is_adj
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_default_dtype(torch.float32)

        self.max_num_sg = max_num_sg
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

        vocab, _1, = self.feature_extractor.get_vocab(*self.train_dataset)
        self.vocab_size = len(vocab)
        self.n_classes = np.unique(self.train_dataset[1]).size

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
        features_sample = []
        adjs_sample = []

        n_sg_max = np.max([len(feature) for feature in features])
        n_sg_used = self.k if self.k > n_sg_max else n_sg_max
        n_sg_used = n_sg_used if n_sg_used < self.max_num_sg else self.max_num_sg
        for i, feature in enumerate(features):
            indices = list(range(len(feature)))
            random.shuffle(indices)
            n_sg_padded = n_sg_used - len(feature)
            extra_indices = [random.choice(indices) for _ in range(n_sg_padded)]
            indices += extra_indices
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

