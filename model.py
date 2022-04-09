import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


class QAGNN(nn.Module):
    def __init__(self, x_init_dim, hid_dim, n_ntype, n_etype, dropout):
        super(QAGNN, self).__init__()
        # self.hid_dim = hid_dim
        # self.n_ntype = n_ntype
        # self.n_etype = n_etype
        # self.dropout = dropout
        self.gnn = GNN(x_init_dim, hid_dim, n_ntype, n_etype, dropout)

    def forward(self, batch):
        return self.gnn(x=batch.x, node_ids=batch.node_ids, node_types=batch.node_types, node_scores=batch.node_scores,
                        edge_index=batch.edge_index, edge_type=batch.edge_type, edge_attr=batch.edge_attr, node2graph=batch.batch)


class GNN(nn.Module):
    def __init__(self, x_init_dim, hid_dim, n_ntype, n_etype, dropout):
        super(GNN, self).__init__()
        self.hid_dim = hid_dim
        self.dropout = dropout
        
        self.relu = nn.ReLU()
        # self.gelu

        self.x2h = nn.Linear(x_init_dim, hid_dim)
        self.ntype_enc = nn.Linear(n_ntype, hid_dim // 2)
        self.nscore_enc = nn.Linear(1, hid_dim // 2)
        
        self.h2h = nn.Linear(2 * hid_dim, hid_dim)

        self.etype_enc = nn.Sequential(
            torch.nn.Linear(n_ntype + n_etype + n_ntype, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            self.relu,
            nn.Linear(hid_dim, hid_dim),
            self.relu
        )

        # self.cmp = CustomMessagePassing(hid_dim, n_ntype, n_etype)
        self.gat_conv = GATConv(hid_dim, hid_dim, add_self_loops=False, edge_dim=hid_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//2),
            self.relu,
            nn.Linear(hid_dim//2, 1)
            # nn.Sigmoid()
        )

    def forward(self, x, node_ids, node_types, node_scores, edge_index, edge_type, edge_attr, node2graph):
        h = self.x2h(x)  # D
        node_types = self.ntype_enc(node_types)  # n_ntype --> D/2
        node_scores = self.nscore_enc(node_scores)  # 1 --> D/2
        
        h = self.h2h(torch.cat([h, node_types, node_scores], dim=-1))  # 2D --> D
        h = self.relu(h)

        edge_attr = self.etype_enc(edge_attr)

        h = self.gat_conv(x=h, edge_index=edge_index, edge_attr=edge_attr)
        
        h = global_mean_pool(h, batch=node2graph)
        h = self.relu(h)

        # h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.mlp(h)

        return h


# class CustomMessagePassing(MessagePassing):
#     def __init__(self, hid_dim, n_ntype, n_etype):
#         super(CustomMessagePassing, self).__init__(aggr='add')
#         self.hid_dim = hid_dim
#         self.n_ntype = n_ntype
#         self.n_etype = n_etype
#
#     def forward(self, x, node_types, node_scores, edge_index, edge_type):
#         return self.propagate(edge_index=edge_index, x=x)
