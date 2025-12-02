
import numpy as np
from typing import Union, List, Callable
import os

import torch 
from torch import Tensor 
from torch import nn 
from torch.nn import init
from torch.nn import LeakyReLU, ReLU, Parameter, BatchNorm1d, Sigmoid

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, VGAE, GATConv, GCNConv, JumpingKnowledge, SAGEConv
from torch_geometric.data import Batch, Data 

from tlayer import GraphTransformerEncoderLayer

from egnn_clean import E_GCL

class RCLG(nn.Module):

    def __init__(
        self,
        gconv_dim: int, 
        tlayer_dim: int,
        in_dim = None, 
        out_dim = None,
        dim_pe = None,
        tlayer_attn_dropout: float = 0., 
        tlayer_ffn_dropout: float = 0., 
        tlayer_ffn_hidden_times: int = 1, 
        gconv_type: str = 'egnn',
        num_layers: int = 10,
        num_heads: int = 4,
        skip_connection: str = 'none',
        norm: str = 'ln',
        out_layer: int = 1, 

    ): 
        super().__init__() 

        self.gconv_dim = gconv_dim 
        self.tlayer_dim = tlayer_dim 

        self.in_dim = in_dim 

        self.dim_pe = dim_pe 

        self.num_layers = num_layers 
        self.skip_connection = skip_connection
        self.se_norm = BatchNorm1d(10)
        self.se_lin = nn.Linear(10, 16)
        self.segment_pooling_fn = None

        self.init_node_edge_encoders()
        self.predict_head = None

        predict_head_modules = []
        last_layer_dim = gconv_dim
        for i in range(out_layer-1):
            predict_head_modules.append(nn.Linear(last_layer_dim, 64))
            predict_head_modules.append(nn.ReLU())
            predict_head_modules.append(nn.Dropout(0.3))
        predict_head_modules.append(nn.Linear(gconv_dim, out_dim))
        self.predict_head = nn.Sequential(*predict_head_modules)

        self.gconvs = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        self.tlayers = nn.ModuleList()
        self.elayers = nn.ModuleList()
        self.etlayer = nn.ModuleList()
        self.middle_layers = nn.ModuleList()
        self.middle_edge_norm = nn.ModuleList()


        for i in range(num_layers):

            if gconv_type == 'gcn':
                self.gconvs.append(GCNConv(gconv_dim, gconv_dim))

            elif gconv_type == 'egnn':
                self.gconvs.append(E_GCL(gconv_dim, gconv_dim, gconv_dim, 0))

            elif gconv_type == 'gat':
                self.gconvs.append(GATConv(in_channels=64, out_channels=16, heads=4, edge_dim=64, dropout=0.2))


            self.middle_layers.append(nn.BatchNorm1d(gconv_dim))

            self.tlayers.append(
                GraphTransformerEncoderLayer(
                    tlayer_dim,
                    num_heads,
                    attn_dropout_ratio=tlayer_attn_dropout,
                    dropout_ratio=tlayer_ffn_dropout,
                    ffn_hidden_times=tlayer_ffn_hidden_times,
                    norm=norm))

            self.middle_edge_norm.append(nn.BatchNorm1d(64))


    def init_node_edge_encoders(self): 

        self.node_encoder = nn.Linear(self.in_dim, self.gconv_dim - self.dim_pe)
        self.edge_encoder = nn.Embedding(1, self.gconv_dim)

    def reset_parameters(self):

        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()

        if self.predict_head:
            for layer in self.predict_head:
                if isinstance(layer, nn.Linear): 
                    layer.reset_parameters() 
            
        for layer in self.gconvs: 
            layer.reset_parameters() 
        for layer in self.tlayers: 
            layer.reset_parameters() 
        for layer in self.middle_layers: 
            if isinstance(layer, nn.BatchNorm1d): 
                layer.reset_parameters() 
            else: 
                for l in layer: 
                    if isinstance(l, nn.Linear) or isinstance(l, nn.LayerNorm):
                        l.reset_parameters()


    def forward(self, data: Union[Data, Batch]):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # edge_attr = data.edge_attr
        pos = data.pos
        x = x.squeeze(-1)
        x = self.node_encoder(x)
        lp = data.se
        lp = self.se_norm(lp)
        lp = self.se_lin(lp)
        x = torch.cat((x, lp), dim=1)
        # edge_attr = self.edge_encoder(edge_attr)
        out = x

        for i in range(self.num_layers):

            x, pos, _ = self.gconvs[i](x, edge_index, pos)


            x = out + x
            x = self.middle_layers[i](x)
            out = x


            graph = (x, batch, edge_index)

            x = self.tlayers[i](graph)


            if self.skip_connection == 'none':
                out = x
            elif self.skip_connection == 'long':
                out = out + x
            elif self.skip_connection == 'short':
                out = out + x
                x = out



        out = self.predict_head(out)

        return out
