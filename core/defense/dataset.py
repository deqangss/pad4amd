import os
import random
import tempfile

import numpy as np
import torch
from scipy.sparse.csr import csr_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

from config import config
from core.droidfeature.feature_extraction import Apk2features
from tools import utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, seed=0, use_cache=False, under_sampling=0.1, device='cuda', feature_ext_args=None):
        """
        build dataset for ml model learning
        :param seed: Integer, the random seed
        :param use_cache: Boolean, cache the representations
        :param under_sampling, Float or None, a positive real-value represents the ratio of benign samples: number_of_mal / under_sampling
        :param device: String, 'cuda' or 'cpu'
        :param feature_ext_args: Dict, arguments for feature extraction
        """
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_default_dtype(torch.float32)
        self.temp_data = utils.SimplifyClass()
        if use_cache:
            self.temp_dir_handle = tempfile.TemporaryDirectory()
            utils.mkdir(self.temp_dir_handle.name)
        else:
            self.temp_dir_handle = None

        self.device = device

        self.feature_ext_args = feature_ext_args
        if feature_ext_args is None:
            self.feature_extractor = Apk2features(config.get('metadata', 'naive_data_pool'),
                                                  config.get('dataset', 'intermediate')
                                                  )
        else:
            assert isinstance(feature_ext_args, dict)
            self.feature_extractor = Apk2features(config.get('metadata', 'naive_data_pool'),
                                                  config.get('dataset', 'intermediate'),
                                                  **feature_ext_args)

        # split the dataset for training, validation, testing
        data_saving_path = os.path.join(config.get('dataset', 'intermediate'), 'dataset.idx')
        if os.path.exists(data_saving_path) and (not self.feature_extractor.update):
            (self.train_dataset, self.validation_dataset, self.test_dataset) = utils.read_pickle(data_saving_path)

            def path_tran(data_paths):
                return np.array(
                    [os.path.join(config.get('metadata', 'naive_data_pool'),
                                  os.path.splitext(os.path.basename(name))[0] + self.feature_extractor.file_ext) for \
                     name in data_paths])

            self.train_dataset = (path_tran(self.train_dataset[0]), self.train_dataset[1])
            self.validation_dataset = (path_tran(self.validation_dataset[0]), self.validation_dataset[1])
            self.test_dataset = (path_tran(self.test_dataset[0]), self.test_dataset[1])
        else:
            mal_feature_paths = self.apk_preprocess(config.get('dataset', 'malware_dir'))
            ben_feature_paths = self.apk_preprocess(config.get('dataset', 'benware_dir'))
            feature_paths = mal_feature_paths + ben_feature_paths
            gt_labels = np.zeros((len(mal_feature_paths) + len(ben_feature_paths)), dtype=np.int32)
            gt_labels[:len(mal_feature_paths)] = 1
            self.train_dataset, self.validation_dataset, self.test_dataset = self.data_split(feature_paths, gt_labels)
            utils.dump_pickle((self.train_dataset, self.validation_dataset, self.test_dataset), data_saving_path)

        print(np.unique(self.train_dataset[1], return_counts=True))
        print(type(self.train_dataset[0]), type(self.train_dataset[1]))
        x_over, y_over = self.random_under_sampling(self.train_dataset[0], self.train_dataset[1], under_sampling=under_sampling)
        print(np.unique(y_over, return_counts=True))
        self.train_dataset = (x_over, y_over)
        import sys
        sys.exit(1)

        self.vocab, _1, _2 = self.feature_extractor.get_vocab(*self.train_dataset)
        self.vocab_size = len(self.vocab)
        self.non_api_size = self.feature_extractor.get_non_api_size(self.vocab)
        self.n_classes = np.unique(self.train_dataset[1]).size

    def data_split(self, feature_paths, labels):
        assert len(feature_paths) == len(labels)
        train_dn, validation_dn, test_dn = None, None, None
        data_split_path = os.path.join(config.get('dataset', 'dataset_dir'), 'tr_te_va_split.name')
        if os.path.exists(data_split_path):
            train_dn, val_dn, test_dn = utils.read_pickle(data_split_path)

        if (train_dn is None) or (validation_dn is None) or (test_dn is None):
            data_names = [os.path.splitext(os.path.basename(path))[0] for path in feature_paths]
            train_dn, test_dn = train_test_split(data_names, test_size=0.2, random_state=self.seed, shuffle=True)
            train_dn, validation_dn = train_test_split(train_dn, test_size=0.25, random_state=self.seed, shuffle=True)
            utils.dump_pickle((train_dn, validation_dn, test_dn),
                              path=data_split_path)

        def query_path(_data_names):
            return np.array(
                [path for path in feature_paths if os.path.splitext(os.path.basename(path))[0] in _data_names])

        def query_indicator(_data_names):
            return [True if os.path.splitext(os.path.basename(path))[0] in _data_names else False for path in
                    feature_paths]

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

    def apk_preprocess(self, apk_paths, labels=None, update_feature_extraction=False):
        old_status = self.feature_extractor.update
        self.feature_extractor.update = update_feature_extraction
        if labels is None:
            feature_paths = self.feature_extractor.feature_extraction(apk_paths)
            self.feature_extractor.update = old_status
            return feature_paths
        else:
            assert len(apk_paths) == len(labels), \
                'uncompilable data shape {} vs. {}'.format(len(apk_paths), len(labels))
            feature_paths = self.feature_extractor.feature_extraction(apk_paths)
            labels_ = []
            for i, feature_path in enumerate(feature_paths):
                fname = os.path.splitext(os.path.basename(feature_path))[0]
                if fname in apk_paths[i]:
                    labels_.append(labels[i])
            self.feature_extractor.update = old_status
            return feature_paths, np.array(labels_)

    def feature_preprocess(self, feature_paths):
        raise NotImplementedError
        # self.feature_extractor.update_cg(feature_paths)

    def feature_api_rpst_sum(self, api_feat_representation_list):
        """
        Summation of api representations
        :param api_feat_representation_list: a list of sparse matrix
        """
        assert isinstance(api_feat_representation_list, list), "Expect a list."
        if len(api_feat_representation_list) > 0:
            assert isinstance(api_feat_representation_list[0], csr_matrix)
        else:
            return np.zeros(shape=(self.vocab_size - self.non_api_size, self.vocab_size - self.non_api_size),
                            dtype=np.float)
        adj_array = np.asarray(api_feat_representation_list[0].todense()).astype(np.float32)
        for sparse_mat in api_feat_representation_list[1:]:
            adj_array += np.asarray(sparse_mat.todense()).astype(np.float32)
        return np.clip(adj_array, a_min=0, a_max=1)

    def get_numerical_input(self, feature_path, label):
        """
        loading features for given a feature path
        # results:
        # --->> mapping feature path to numerical representations
        # --->> features: 1d array, and a list of sparse matrices
        # --->> label: scalar
        """
        feature_vector, label = self.feature_extractor.feature2ipt(feature_path, label,
                                                                   self.vocab,
                                                                   self.temp_dir_handle)
        return feature_vector, label

    def get_numerical_input_batch(self, feature_paths, labels, name='train'):
        rpst_saving_path = os.path.join(config.get('dataset', 'intermediate'), '{}.npz'.format(name))
        if not os.path.exists(rpst_saving_path):
            X1, X2 = [], []
            for feature_path, label in zip(feature_paths, labels):
                non_api_rpst, api_rpst, label = self.get_numerical_input(feature_path, label)
                X1.append(non_api_rpst)
                X2.append(api_rpst)
            utils.dump_pickle((X1, X2, labels), rpst_saving_path, use_gzip=True)
            return X1, X2, labels
        else:
            return utils.read_pickle(rpst_saving_path, use_gzip=True)

    def get_input_producer(self, feature_paths, y, batch_size, name='train'):
        params = {'batch_size': batch_size,
                  'num_workers': self.feature_ext_args['proc_number'],
                  'shuffle': False}
        return torch.utils.data.DataLoader(DatasetTorch(feature_paths, y, self, name=name),
                                           worker_init_fn=lambda x: np.random.seed(
                                               torch.randint(0, 2 ^ 31, [1, ])[0] + x),
                                           **params)

    @staticmethod
    def random_under_sampling(X, y, under_sampling=None):
        """
        under sampling
        :param X: data
        :type 1D numpy array
        :param y: label
        :type 1D numpy.ndarray
        :param ratio: proportion
        :type float
        :return: X, y
        """
        if under_sampling is None:
            return X, y
        if not isinstance(under_sampling, float):
            raise TypeError("{}".format(type(under_sampling)))
        if under_sampling > 1.:
            ratio = 1.
        if under_sampling < 0.:
            ratio = 0.

        if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
            raise TypeError

        _labels, _counts = np.unique(y, return_counts=True)
        _label_count = dict(zip(_labels, _counts))
        _mal_count, _ben_count = _label_count[1], _label_count[0]
        _ratio = _mal_count / float(_ben_count)
        if _ratio >= under_sampling:
            return X, y

        _ben_count = int(_mal_count / under_sampling)
        random_indices = np.random.choice(
            np.where(y == 1)[0], _ben_count, replace=False
        )
        ben_x = X[random_indices]
        ben_y = y[random_indices]
        mal_x = X[y==0]
        mal_y = y[y==0]
        X = np.vstack([mal_x, ben_x])
        y = np.vstack([mal_y, ben_y])
        np.random.seed(0)
        np.random.shuffle(X)
        np.random.seed(0)
        np.random.shuffle(y)
        return X, y

    def clear_up(self):
        self.temp_data.cleanup()

    @staticmethod
    def get_modification(adv_x, x, idx, sp=True):
        assert isinstance(adv_x, (np.ndarray, torch.Tensor))
        assert isinstance(x, (np.ndarray, torch.Tensor))
        x_mod = adv_x - x
        if isinstance(x_mod, np.ndarray):
            x_mod = np.array([x_mod[i, idx[i]] for i in range(x.shape[0])])
        else:
            x_mod = torch.stack([x_mod[i, idx[i]] for i in range(x.shape[0])])

        if sp:
            if isinstance(x_mod, torch.Tensor):
                return x_mod.to_sparse().cpu().unbind(dim=0)
            else:
                return torch.tensor(x_mod, dtype=torch.int).to_sparse().cpu().unbind(dim=0)
        else:
            if isinstance(x_mod, torch.Tensor):
                return x_mod.cpu().unbind(dim=0)
            else:
                return np.split(x_mod, x_mod.shape[0], axis=0)

    @staticmethod
    def modification_integ(x_mod_integrated, x_mod):
        assert isinstance(x_mod_integrated, list) and isinstance(x_mod, list)
        if len(x_mod_integrated) == 0:
            return x_mod
        assert len(x_mod_integrated) == len(x_mod)
        for i in range(len(x_mod)):
            # warning: the addition is list appending when tensors are on gpu,
            # while it is summation of two tensors on cpu
            assert not x_mod[i].is_cuda
            x_mod_integrated[i] += x_mod[i]
        return x_mod_integrated


class DatasetTorch(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, feature_paths, datay, dataset_obj, name='train'):
        'Initialization'
        try:
            assert (name == 'train' or name == 'test' or name == 'val')
        except Exception as e:
            raise AssertionError("Only support selections: 'train', 'val' or 'test'.\n")

        self.feature_paths = feature_paths
        self.datay = datay
        self.dataset_obj = dataset_obj
        self.name = name

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.feature_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.get_item(index)

    def get_item(self, index):
        if len(self.dataset_obj.temp_data.cache_arr) > index:
            return self.dataset_obj.temp_data.cache_arr[index], self.datay[index]
        else:
            return self.dataset_obj.get_numerical_input(self.feature_paths[index], self.datay[index])
