import torch
import torch.nn as nn
import torch.nn.functional as F

from core.defense.layers import GraphAttentionLayer, SpGraphAttentionLayer, GraphAttentionLayerCLS


class MalGAT(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 n_hidden_units,
                 penultimate_hidden_unit,
                 n_heads,
                 dropout,
                 alpha,
                 k,
                 sparse,
                 activation=F.elu):
        """
        Graph ATtention networks for malware detection
        :param vocab_size: Integer, the number of words in the  dictionary
        :param embedding_dim: Integer, the number of embedding codes
        :param n_hidden_units: List, a list of integers denote the number of neurons of hidden layers
        :param penultimate_hidden_unit: Integer, the number of neurons in the penultimate layer
        :param n_heads: Integer, the number of headers to learn a sub-graph
        :param dropout: Float, dropout rate applied to attention layer
        :param alpha: Float, the slope coefficient of leaky-relu
        :param k: Integer, the sampling size
        :param sparse: GAT in sparse version or not
        :param activation: activation function
        """
        super(MalGAT, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden_units = n_hidden_units
        assert (isinstance(self.n_hidden_units, list)) & (len(self.n_hidden_units) > 0)
        self.penultimate_hidden_unit = penultimate_hidden_unit
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.k = k
        self.sparse = sparse
        self.activation = activation

        # instantiated trainable parameters (layers)
        self.embedding_weight = nn.Parameter(torch.empty(size=(self.vocab_size, self.embedding_dim)))
        nn.init.normal_(self.embedding_weight.data)  # default initialization method in torch

        graph_attn_layer = GraphAttentionLayer if not sparse else SpGraphAttentionLayer

        self.attn_layers = []
        for pre_unit, current_unit in zip([self.embedding_dim] + self.n_hidden_units[:-1], self.n_hidden_units):
            attn_headers = []
            if len(self.attn_layers) <= 0:
                feature_in = pre_unit
            else:
                feature_in = pre_unit * self.n_heads
            for head_id in range(self.n_heads):
                attn_headers.append(graph_attn_layer(feature_in,
                                                     current_unit,
                                                     self.dropout,
                                                     self.alpha,
                                                     concat=True))
            self.attn_layers.append(attn_headers)
        # registration
        for idx_i, attn_headers in enumerate(self.attn_layers):
            for idx_j, header in enumerate(attn_headers):
                self.add_module('attention_layer_{}_header_{}'.format(idx_i, idx_j), header)

        self.cls_weight = nn.Parameter(torch.empty(size=(self.n_hidden_units[-1] * self.n_heads,)))
        nn.init.normal_(self.cls_weight.data)
        self.cls_attn_layer = GraphAttentionLayerCLS(self.n_hidden_units[-1] * self.n_heads,
                                                     self.dropout,
                                                     self.alpha)

        # self.dense = nn.Linear(self.n_hidden_units[-1] * self.n_heads, self.penultimate_hidden_unit)
        self.dense = nn.Linear(self.vocab_size, self.penultimate_hidden_unit)

    def forward(self, x, adjs=None):
        """
        forward the neural network
        :param x: 3d tensor,  feature representations in the mini-batch level, [self.k, batch_size, vocab_dim]
        :param adjs: 4d tensor, adjacent matrices in the mini-batch level, [self.k, batch_size, vocab_dim, vocab_dim]
        :return: None
        """
        assert len(x) == self.k and self.k > 0
        # features
        embed_features = torch.stack(
            [self.embedding_weight] * x.size()[1])  # embed_features shape is [batch_size, vocab_size, vocab_dim]
        assert len(x) == self.k  # x has the shape [self.k, batch_size, vocab_size]

        # x_comb = torch.clip(torch.sum(x, dim=0), min=0, max=1.)
        # if adjs is None:
        #     if self.sparse:
        #         if self.training:
        #             adjs = torch.stack([
        #                 torch.stack([torch.matmul(_x_e.unsqueeze(-1), _x_e.unsqueeze(0)).to_sparse() for _x_e in _x]) \
        #                 for _x in x
        #             ])
        #         else:
        #             adjs = torch.stack([torch.matmul(_x.unsqueeze(-1), _x.unsqueeze(-2)).to_sparse() for _x in x])
        #     else:
        #         adjs = torch.stack([torch.matmul(_x.unsqueeze(-1), _x.unsqueeze(-2)) for _x in x])
        # if adjs.is_sparse:
        #     adj = torch.sparse.sum(adjs, dim=0)
        #     if not self.sparse:
        #         adj = adj.to_dense()
        # else:
        #     adj = torch.sum(adjs, dim=0)
        #
        # features = torch.unsqueeze(x_comb, dim=-1) * embed_features
        # for headers in self.attn_layers:
        #     features = F.dropout(features, self.dropout, training=self.training)
        #     features = torch.cat([header(features, adj) for header in headers], dim=-1)
        #
        # # latent_codes = [torch.stack([self.cls_weight] * x[0].size()[0])]
        # latent_codes = []
        # for i in range(self.k):
        #     latent_code = torch.unsqueeze(x[i], dim=-1) * features  # masking out the unused representations via broadcasting, herein the latent_code shape is [batch_size, vocab_size, feature_dim]
        #     latent_code = torch.amax(latent_code, dim=1)  # after max pooling, the latent_code shape is [batch_size, feature_dim]
        #     latent_codes.append(latent_code)

        # latent_codes = torch.stack(latent_codes, dim=1)  # the result shape is [batch_size, self.k+1, feature_dim]
        # # latent_codes = self.cls_attn_layer(latent_codes)
        # latent_codes = torch.amax(latent_codes, dim=1)
        # latent_codes = self.activation(self.dense(latent_codes))
        latent_codes = []
        for i in range(self.k):
            latent_codes.append(self.activation(self.dense(x[i])))
        latent_codes = torch.amax(torch.stack(latent_codes, dim=1), dim=1)
        # latent_codes = self.activation(self.dense(x_comb))
        return latent_codes
