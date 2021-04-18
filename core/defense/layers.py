"""
code is adapted from the github repository: https://github.com/Diego999/pyGAT
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayerCLS(nn.Module):
    def __init__(self, feature_dim, dropout, alpha, smooth=False):
        super(GraphAttentionLayerCLS, self).__init__()
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.alpha = alpha
        if not smooth:
            self.activation = nn.LeakyReLU(self.alpha)
        else:
            self.activation = nn.ELU(self.alpha)

        self.a = nn.Parameter(torch.empty(size=(2 * self.feature_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, cls):

        N = h.size()[1] # h.shape = (batch_size, number_of_subgraphs, feature_dim)
        cls_h_repeated = cls.unsqueeze(dim=1).repeat_interleave(N, dim=1)
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN

        all_combinations_matrix = torch.cat([cls_h_repeated, h], dim=-1)  # shape is [batch_size, number_of_subgraphs, 2 * feature_dim]
        all_combinations_matrix.size()
        attention = self.activation(torch.matmul(all_combinations_matrix, self.a)).permute(0, 2, 1) # attention.shape is [batch_size, 1, number_of_subgraphs+1]
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.squeeze(torch.matmul(attention, h), dim=1)
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.feature_dim) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, smooth=False, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.smooth = smooth
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if not self.smooth:
            self.activation = nn.LeakyReLU(self.alpha)
        else:
            self.activation = nn.ELU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (batch_size, vocab_size, in_features), Wh.shape: (batch_size, vocab_size, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.activation(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # vocab size

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        # all_combinations_matrix.shape == (batch_size, N * N, 2 * out_features), RAM consuming

        return all_combinations_matrix.view(-1, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, smooth=False, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout
        self.smooth = smooth
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        if not self.smooth:
            self.activation = nn.LeakyReLU(self.alpha)
        else:
            self.activation = nn.ELU(self.alpha)

    def forward(self, input, adj):
        """
        pass through the sparse graph attention layer
        :param input: 3D tensor [batch_size, node_number, feature_dim]
        :param adj: 3D tensor [batch_size, node_number, node_number]
        :return:
        """
        dv = 'cuda' if input.is_cuda else 'cpu'
        assert input.size()[0] == adj.size()[0]
        batch_size = adj.size()[0]
        N = input.size()[1]
        if adj.is_sparse:
            edge = adj._indices().to(dv)
        else:
            edge = adj.nonzero(as_tuple=False).t()

        if input.is_sparse:
            h = torch.stack([torch.sparse.mm(input_e, self.W) for input_e in input])
        else:
            h = torch.matmul(input, self.W)

        # h: N x out
        assert not torch.isnan(h).any()
        for _h in h:
            print(_h.shape)
        print(edge.shape)
        for e in edge:
            print(e.shape)
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], edge[1, :], :], h[edge[0, :], edge[2, :], :]), dim=1).t()
        # edge: 2*D x E
        print(edge_h.shape)

        edge_e = torch.exp(-self.activation(self.a.matmul(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        sp_edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([batch_size, N, N]))
        e_rowsum = torch.stack(
            [torch.sparse.mm(sp_e, torch.ones(size=(N, 1), dtype=torch.float, device=dv)) for sp_e in sp_edge_e])
        # e_rowsum: batch_size x N x 1

        edge_e = F.dropout(edge_e, self.dropout, training=self.training)
        # edge_e: E

        sp_edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([batch_size, N, N]))
        h_prime = torch.stack(
            [torch.sparse.mm(sp_e, dense_e) for sp_e, dense_e in zip(sp_edge_e, h)]
        )
        # h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: batch_size x N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: batch_size x N x out
        h_prime = torch.where(torch.isnan(h_prime), torch.zeros_like(h_prime), h_prime)  # masking out isolated nodes (i.e., no edge connections to other nodes)
        # assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
