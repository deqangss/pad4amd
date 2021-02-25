import os
import signal
import warnings
import joblib
import pickle as pkl

import scipy.sparse as sp
import numpy as np
import torch

#################################################################################
################################# smali code ####################################
#################################################################################

def java_class_name2smali_name(cls):
    """
       Transform a typical xml format class into smali format

       :param cls: the input class name
       :rtype: string
    """
    if cls is None:
        return
    if not isinstance(cls, str):
        raise ValueError("Expected a string")

    return "L" + cls.replace(".", "/") + ";"


def remove_duplicate(components):
    if isinstance(components, list):
        return ['.'.join(list(filter(None, comp.strip().split('.')))) for comp in components]
    elif isinstance(components, str):
        return '.'.join(list(filter(None, components.strip().split('.'))))
    else:
        raise TypeError("Types of 'list' and 'str' are expected, but got {}.".format(type(components)))


def pool_initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def retrive_files_set(base_dir, dir_ext, file_ext):
    """
    get file paths given the directory
    :param base_dir: basic directory
    :param dir_ext: directory append at the rear of base_dir
    :param file_ext: file extension
    :return: set of file paths. Avoid the repetition
    """

    def get_file_name(root_dir, file_ext):

        for dir_path, dir_names, file_names in os.walk(root_dir, topdown=True):
            for file_name in file_names:
                _ext = file_ext
                if os.path.splitext(file_name)[1] == _ext:
                    yield os.path.join(dir_path, file_name)
                elif '.' not in file_ext:
                    _ext = '.' + _ext

                    if os.path.splitext(file_name)[1] == _ext:
                        yield os.path.join(dir_path, file_name)
                    else:
                        pass
                else:
                    pass

    if file_ext is not None:
        file_exts = file_ext.split("|")
    else:
        file_exts = ['']
    file_path_list = list()
    for ext in file_exts:
        file_path_list.extend(get_file_name(os.path.join(base_dir, dir_ext), ext))
    # remove duplicate elements
    from collections import OrderedDict
    return list(OrderedDict.fromkeys(file_path_list))


def check_dir(sample_dir):
    """
    check a valid directory and produce a list of file paths
    """
    if isinstance(sample_dir, str):
        if not os.path.exists(sample_dir):
            MSG = "No such directory or file {} exists!".format(sample_dir)
            raise ValueError(MSG)
        elif os.path.isfile(sample_dir):
            sample_path_list = [sample_dir]
        elif os.path.isdir(sample_dir):
            sample_path_list = list(retrive_files_set(sample_dir, "", ".apk|"))
            if len(sample_path_list) <= 0:
                warnings.warn('No files')
        else:
            raise ValueError(" No such path {}".format(sample_dir))
    elif isinstance(sample_dir, list):
        sample_path_list = [path for path in sample_dir if os.path.isfile(path)]
    else:
        MSG = "A directory or a list of paths are allowed!"
        raise ValueError(MSG)

    return sample_path_list


def dump_joblib(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    try:
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def read_joblib(path):
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def dump_pickle(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    with open(path, 'wb') as wr:
        pkl.dump(data, wr)
    return True


def read_pickle(path):
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return pkl.load(fr)
    else:
        raise IOError("The {0} is not been found.".format(path))


def dump_list_of_lists(data, path):
    assert isinstance(data[0], list) and isinstance(data, list)
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    with open(path, 'wb') as f:
        for inter_list in data:
            f.write((','.join(inter_list) + '\n').encode())
    return


def read_list_of_lists(path):
    assert os.path.exists(path) and os.path.isfile(path)

    rtn_data = []
    with open(path, 'rb') as f:
        for l in f.readlines():
            rtn_data.append(l.decode('utf-8').strip().split(','))


def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        raise Exception("Fail to create directory! Error:" + str(e))


def read_txt(path, mode='r'):
    if os.path.isfile(path):
        with open(path, mode) as f_r:
            lines = f_r.read().strip().splitlines()
            return lines
    else:
        raise ValueError("{} does not seen like a file path.\n".format(path))


def dump_txt(data_str, path, mode='w'):
    if not isinstance(data_str, str):
        raise TypeError

    with open(path, mode) as f_w:
        f_w.write(data_str)


def build_kwargs(keys, arg_dict):
    st = ''
    for key in keys:
        st += '%s:%s\n' % (key, str(arg_dict[key]))
    return st


def save_args(fout, args):
    if isinstance(args, str):
        dump_txt(args, fout, mode='w')
    elif isinstance(args, dict):
        args_str = build_kwargs(args.keys(), args)
        dump_txt(args_str, fout, mode='w')
    else:
        raise TypeError("Expected str or dict.")


def get_group_args(args, args_parser, title):
    import argparse
    assert isinstance(args, argparse.Namespace) and isinstance(args_parser, argparse.ArgumentParser)
    for group in args_parser._action_groups:
        if group.title == title:
            return {action.dest: getattr(args, action.dest, None) for action in group._group_actions}
        else:
            continue
    return


def tensor_coo_sp_to_ivs(sparse_tensor):
    return sparse_tensor._indices(), sparse_tensor._values(), sparse_tensor.size()


def ivs_to_tensor_coo_sp(ivs, device = 'cpu'):
    return torch.sparse_coo_tensor(ivs[0], ivs[1], ivs[2], device=device)


def sp_to_symmetric_sp(sparse_mx):
    sparse_mx = sparse_mx + sparse_mx.T.multiply(sparse_mx.T > sparse_mx) - sparse_mx.multiply(sparse_mx.T > sparse_mx)
    sparse_eye = sp.csr_matrix(sparse_mx.sum(axis=0) > 1e-8).T.multiply(sp.eye(sparse_mx.shape[0]))
    return (sparse_mx.multiply(1. - np.eye(*sparse_eye.shape))).tocsr() + sparse_eye


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


def to_tensor(features=None, adj=None, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.
    code is adapted from: https://github.com/deeprobust/DeepRobust/graph/utils.py
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """

    def _to_torch_tensor(mat):
        if isinstance(mat, tuple):
            mat = ivs_to_tensor_coo_sp(mat, device=device)
        if sp.issparse(mat):
            mat = sparse_mx_to_torch_sparse_tensor(mat)
        elif isinstance(mat, torch.Tensor):
            pass
        else:
            mat = torch.FloatTensor(mat)
        return mat

    features = _to_torch_tensor(features).to(device)
    if adj is not None:
        adj = _to_torch_tensor(adj)
    if labels is None:
        return features, adj
    else:
        labels = torch.LongTensor(labels).to(device)
        return features, adj, labels


def to_tensors(features, adj=None, labels=None, device='cpu'):
    """
    Sample as 'to_tensor' but work on list of features
    """
    assert isinstance(features, list)
    if adj is not None:
        assert len(features) == len(adj)
    if adj is None:
        for i in range(len(features)):
            features[i], _ = to_tensor(features[i], adj, device=device)
    else:
        for i in range(len(features)):
            features[i], adj[i] = to_tensor(features[i], adj[i], device=device)

    if labels is None:
        return features, adj
    else:
        labels = torch.LongTensor(labels).to(device)
        return features, adj, labels
