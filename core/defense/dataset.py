import os
import random
import tempfile

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from config import config
from core.droidfeature.feature_extraction import Apk2graphs
from tools import utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, k=8, is_adj=False, seed=0, n_sgs_max=1000, feature_ext_args=None):
        """
        build dataset for ml model learning
        :param k: Integer, the number of subgraphs is sampled for passing through the neural networks
        :param is_adj: Boolean, whether use the actual adjacent matrix or not
        :param seed: Integer, the random seed
        :param n_sgs_max: Integer, the maximum number of subgraphs
        :param feature_ext_args: Dict, arguments for feature extraction
        """
        self.k = k
        self.is_adj = is_adj
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_default_dtype(torch.float32)
        assert self.k < n_sgs_max
        self.n_sgs_max = n_sgs_max
        self.feature_ext_args = feature_ext_args
        self.temp_dir_handle = tempfile.TemporaryDirectory()
        if feature_ext_args is None:
            self.feature_extractor = Apk2graphs(config.get('metadata', 'naive_data_pool'),
                                                config.get('dataset', 'intermediate'))
        else:
            assert isinstance(feature_ext_args, dict)
            self.feature_extractor = Apk2graphs(config.get('metadata', 'naive_data_pool'),
                                                config.get('dataset', 'intermediate'),
                                                **feature_ext_args)

        data_saving_path = os.path.join(config.get('dataset', 'intermediate'), 'dataset.idx')
        if os.path.exists(data_saving_path):
            (self.train_dataset, self.validation_dataset, self.test_dataset) = utils.read_pickle(data_saving_path)

            def path_tran(data_paths):
                return np.array(
                    [os.path.join(config.get('metadata', 'naive_data_pool'), os.path.basename(name)) for name in data_paths])
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

        _labels, counts = np.unique(self.train_dataset[1], return_counts=True)
        self.sample_weights = np.ones_like(_labels).astype(np.float32)
        _weights = float(np.max(counts)) / counts
        for i in range(_labels.shape[0]):
            self.sample_weights[_labels[i]] = _weights[i]

        vocab, _1, flag = self.feature_extractor.get_vocab(*self.train_dataset)
        self.vocab_size = len(vocab)
        self.n_classes = np.unique(self.train_dataset[1]).size
        if flag:
            self.feature_extractor.update_cg(self.train_dataset[0])
            self.feature_extractor.update_cg(self.validation_dataset[0])
            self.feature_extractor.update_cg(self.test_dataset[0])

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

    def get_numerical_input(self, feature_paths, labels):
        """
        loading features for given a list of feature paths
        # results:
        # --->> mapping feature paths to numerical representations
        # --->> features: 2d list [number of files, number of subgraphs], in which each element
        # has a vector with size [vocab_size]
        # --->> _labels: 1d list [number of files]
        # --->> adjs: 2d list [number of files, number of subgraphs], in which each element has
        # a scipy sparse matrix with size [vocab_size, vocab_size]
        """
        return self.feature_extractor.feature2ipt(feature_paths, labels, self.is_adj, self.n_sgs_max)

    def collate_fn(self, batch):
        # 1. Because the number of sub graphs is different between apks, we here align a batch of data
        # pad the subgraphs if an app has subgraph smaller than self.k
        # 2. We change the sparse adjacent matrix to its tuple of (indices, values, shape), accommodating the
        # unsupported issue of dataloader
        features = [item[0] for item in batch]
        adjs = [item[1] for item in batch]
        labels_ = [item[2] for item in batch]

        batch_size = len(features)
        features_padded = []
        adjs_padded = []
        g_ind = []

        import time
        start_time = time.time()
        batch_n_sg_max = np.max([len(feature) for feature in features])
        n_sg_used = batch_n_sg_max if batch_n_sg_max < self.n_sgs_max else self.n_sgs_max
        n_sg_used = n_sg_used if n_sg_used > self.k else self.k
        for i, feature in enumerate(features):
            is_padding = True if len(feature) < n_sg_used else False
            if not is_padding:
                indices = np.random.choice(len(feature), n_sg_used, replace=False)
                features_padded.append([feature[_i] for _i in indices])
                if self.is_adj:
                    adjs_padded.append([adjs[i][_i] for _i in indices])
                indices_slicing = np.array(list(map(dict(zip(indices, range(n_sg_used))).get, range(n_sg_used))))
            else:
                n = n_sg_used - len(feature)
                indices = np.arange(n_sg_used)
                feature.extend([np.zeros_like((feature[0]), dtype=np.float32) for _ in range(n)])
                features_padded.append(feature)
                if self.is_adj:
                    adjs[i].extend([csr_matrix(adjs[i][0].shape, dtype=np.float32) for _ in range(n)])
                    adjs_padded.append(adjs[i])
                indices_slicing = indices
            g_ind.append(indices_slicing)

        # shape [batch_size, self.n_sg_used, vocab_size]
        features_padded = np.array([np.stack(list(feat), axis=0) for feat in zip(*features_padded)]).transpose(1, 0, 2)

        if self.is_adj:
            # A list (with size self.k) of sparse adjacent matrix in the mini-batch level, in which each element
            # has the shape [batch_size, vocab_size, vocab_size]
            for i in range(batch_size):
                for j in range(self.k):
                    adjs_padded[i][j] = utils.sparse_mx_to_torch_sparse_tensor(
                        utils.sp_to_symmetric_sp(adjs_padded[i][j])
                    )
            adjs_padded_t = torch.stack([torch.stack(list(adj)) for adj in list(zip(*adjs_padded))[:self.k]])
            # dataloader does not support sparse matrix
            adjs_padded_tuple = utils.tensor_coo_sp_to_ivs(adjs_padded_t)
        else:
            adjs_padded_tuple = None
        print('processing time:', time.time() - start_time)
        return features_padded, adjs_padded_tuple, labels_, np.array(g_ind)

    def get_input_producer(self, data, y, batch_size, name='train'):
        params = {'batch_size': batch_size,
                  'num_workers': self.feature_ext_args['proc_number'],
                  'collate_fn': self.collate_fn,
                  'shuffle': False}
        return torch.utils.data.DataLoader(DatasetTorch(data, y, self, name=name),
                                           worker_init_fn=lambda x: np.random.seed(torch.randint(0, 2^31, [1,])[0] + x),
                                           **params)

    @staticmethod
    def get_modification(adv_x, x, g_ind, sp=True):
        assert isinstance(adv_x, (np.ndarray, torch.Tensor))
        assert isinstance(x, (np.ndarray, torch.Tensor))
        x_mod = adv_x - x
        if isinstance(x_mod, np.ndarray):
            x_mod = np.array([x_mod[i, g_ind[i]] for i in range(x.shape[0])])
        else:
            x_mod = torch.stack([x_mod[i, g_ind[i]]for i in range(x.shape[0])])

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
        import time
        start_time = time.time()
        x, adj, y = self.dataset_obj.get_numerical_input([feature_path], [y])
        print('loading time:', time.time() - start_time)
        assert len(x) > 0 and len(adj) > 0, "Fail to load: " + feature_path
        return x[0], adj[0], y[0]

