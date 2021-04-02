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
                 alpha_,
                 k,
                 use_fusion,
                 sparse,
                 smooth=False,
                 activation=F.elu):
        """
        Graph ATtention networks for malware detection
        :param vocab_size: Integer, the number of words in the  dictionary
        :param embedding_dim: Integer, the number of embedding codes
        :param n_hidden_units: List, a list of integers denote the number of neurons of hidden layers
        :param penultimate_hidden_unit: Integer, the number of neurons in the penultimate layer
        :param n_heads: Integer, the number of headers to learn a sub-graph
        :param dropout: Float, dropout rate applied to attention layer
        :param alpha_: Float, the slope coefficient of leaky-relu or elu
        :param k: Integer, the sampling size
        :param use_fusion: Boolean, combing the graph-type feature and binary bag-of-words feature
        :param sparse: GAT in sparse version or not
        :param smooth: replace lrelu of elu
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
        self.alpha = alpha_
        self.k = k
        self.use_fusion = use_fusion
        self.sparse = sparse
        self.smooth = smooth
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
                                                     smooth=self.smooth,
                                                     concat=True))
            self.attn_layers.append(attn_headers)
        # registration
        for idx_i, attn_headers in enumerate(self.attn_layers):
            for idx_j, header in enumerate(attn_headers):
                self.add_module('attention_layer_{}_header_{}'.format(idx_i, idx_j), header)

        self.attn_out = graph_attn_layer(self.n_hidden_units[-1] * self.n_heads,
                                         penultimate_hidden_unit,
                                         self.dropout,
                                         self.alpha,
                                         smooth=self.smooth,
                                         concat=True)
        self.add_module('attention_layer_out', self.attn_out)

        self.cls_attn_layers = []
        for head_id in range(self.n_heads):
            self.cls_attn_layers.append(
                GraphAttentionLayerCLS(self.penultimate_hidden_unit,
                                       self.dropout,
                                       self.alpha)
            )
        # registration
        for idx_i, cls_attn_layer in enumerate(self.cls_attn_layers):
            self.add_module('attention_cls_layer_header_{}'.format(idx_i), cls_attn_layer)

        self.attn_dense = nn.Linear(self.vocab_size, self.embedding_dim)
        self.cls_weight = nn.Parameter(torch.empty(size=(self.penultimate_hidden_unit, )))
        nn.init.normal_(self.cls_weight.data)
        # self.mod_gra_cls_dense = nn.Linear(self.penultimate_hidden_unit, self.penultimate_hidden_unit, bias=False)

        # another modality function
        self.mod_frq_dense = nn.Linear(self.vocab_size, self.embedding_dim)
        self.mod_frq_cls_dense = nn.Linear(self.embedding_dim, self.penultimate_hidden_unit)

    def forward(self, x, adjs=None):
        """
        forward the neural network
        :param x: 3d tensor,  feature representations in the mini-batch level, [batch_size, self.k, vocab_dim]
        :param adjs: 4d tensor, adjacent matrices in the mini-batch level, note that shape is [self.k, batch_size,  vocab_dim, vocab_dim] because sparse tensor has no stride
        :return: None
        """
        x = x.permute(1, 0, 2)  # [batch_size, k, vocab_dim] --> [k, batch_size, vocab_dim]
        assert (len(x) >= self.k) and (self.k >= 0)  # x has the shape [self.k, batch_size, vocab_size]
        x_comb = torch.clip(torch.sum(x, dim=0), min=0, max=1.)
        mod1_code = torch.amax(
            self.activation(self.mod_frq_dense((x_comb.unsqueeze(-1) * self.embedding_weight).permute(0, 2, 1))), dim=-1)
        if self.k <= 0:
            return self.activation(self.mod_frq_cls_dense(mod1_code))

        if adjs is None:
            if self.sparse:
                adjs = torch.stack([
                    torch.stack([torch.matmul(_x_e.unsqueeze(-1), _x_e.unsqueeze(0)).to_sparse() for _x_e in _x]) \
                    for _x in x[:self.k]
                ])
            else:
                adjs = torch.stack([torch.matmul(_x.unsqueeze(-1), _x.unsqueeze(-2)) for _x in x[:self.k]])

        latent_codes = []
        for i in range(self.k):
            features = torch.unsqueeze(x[i], dim=-1) * torch.unsqueeze(self.embedding_weight, dim=0)
            for headers in self.attn_layers:
                features = F.dropout(features, self.dropout, training=self.training)
                features = torch.cat([header(features, adjs[i]) for header in headers], dim=-1)
            features = F.dropout(features, self.dropout, training=self.training)
            features = self.attn_out(features, adjs[i])
            attn_code = torch.amax(
                self.activation(self.attn_dense((x[i].unsqueeze(-1) * features).permute(0, 2, 1))), dim=-1)
            latent_codes.append(attn_code)
        latent_codes = torch.stack(latent_codes, dim=1)  # latent_codes: [batch_size, self.k, feature_dim]
        latent_codes = F.dropout(latent_codes, self.dropout, training=self.training)
        cls_code = torch.stack([self.cls_weight] * x[0].size()[0])
        # cls_code = self.activation(self.mod_frq_cls_dense(mod1_code))
        if self.use_fusion:
            latent_codes = self.activation(torch.stack([header_cls(latent_codes, cls_code) for header_cls in self.cls_attn_layers], dim=-2).sum(
                    -2) / self.n_heads + self.mod_frq_cls_dense(mod1_code))
        else:
            latent_codes = self.activation(
                torch.stack([header_cls(latent_codes, cls_code) for header_cls in self.cls_attn_layers], dim=-2).sum(
                    -2) / self.n_heads)
        return latent_codes
