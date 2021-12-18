import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.
    Examples
    --------
	We can first load dataset and then train GCN.
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gcn = gcn.to('cpu')
    >>> gcn.fit(features, adj, labels, idx_train) # train without earlystopping
    >>> gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with earlystopping
    >>> gcn.test(idx_test)
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5,
                 with_relu=True, with_bias=True, smooth=True,
                 alpha_=0.2, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.smooth = smooth
        self.alpha_ = alpha_
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        if self.with_relu and (not self.smooth):
            x = F.relu(self.gc1(x, adj))
        elif self.with_relu and self.smooth:
            x = F.elu(self.gc1(x, adj), alpha=self.alpha_)
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.with_relu and (not self.smooth):
            x = F.relu(self.gc2(x, adj))
        elif self.with_relu and self.smooth:
            x = F.elu(self.gc2(x, adj), alpha=self.alpha_)
        else:
            x = self.gc2(x, adj)

        return torch.mean(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
