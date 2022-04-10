import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from transformers import AutoModel, BertModel


class QAGNN(nn.Module):
    def __init__(self, lm_name, hid_dim, n_ntype, n_etype, dropout):
        super(QAGNN, self).__init__()

        self.text_enc = AutoModel.from_pretrained(lm_name, output_hidden_states=True)
        for param in self.text_enc.base_model.parameters():
            param.requires_grad = False

        self.qa_dim = self.text_enc.config.hidden_size
        self.gnn = GNN(self.qa_dim, hid_dim, n_ntype, n_etype, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.qa_dim + 2 * hid_dim, 2 * hid_dim),
            nn.ReLU(),
            nn.Linear(2 * hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, 1)
            # nn.Sigmoid()
        )

    def forward(self, batch):

        text_hid_states = self.text_enc(input_ids=batch.input_ids, token_type_ids=batch.segment_ids, attention_mask=batch.input_mask)
        last_hid_states = text_hid_states[-1][-1]
        qa_emb = last_hid_states.mean(dim=0)  # (qa_dim,)

        qa_node_emb, pooled_graph_emb = self.gnn(qa_emb=qa_emb, x=batch.x, node_ids=batch.node_ids, node_types=batch.node_types, node_scores=batch.node_scores,
                        edge_index=batch.edge_index, edge_type=batch.edge_type, edge_attr=batch.edge_attr, node2graph=batch.batch)  # (hid_dim,), (hid_dim,)

        emb = torch.concat([qa_emb, qa_node_emb, pooled_graph_emb], dim=-1)
        emb = self.mlp(emb)

        return emb


class GNN(nn.Module):
    def __init__(self, qa_dim, hid_dim, n_ntype, n_etype, dropout):
        super(GNN, self).__init__()
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        # self.gelu

        self.qa2node = nn.Linear(qa_dim, hid_dim)
        # self.x2h = nn.Linear(x_init_dim, hid_dim)
        self.ntype_nscore_enc = nn.Linear(n_ntype + 1, hid_dim // 2)

        # self.h2h = nn.Linear(2 * hid_dim, hid_dim)

        self.etype_enc = nn.Sequential(
            torch.nn.Linear(n_ntype + n_etype + n_ntype, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            self.act,
            nn.Linear(hid_dim, hid_dim),
            self.act
        )

        gc_hid_dim = hid_dim + hid_dim // 2
        self.gat_conv = GATConv(gc_hid_dim, hid_dim, add_self_loops=False, edge_dim=hid_dim)

    def forward(self, qa_emb, x, node_ids, node_types, node_scores, edge_index, edge_type, edge_attr, node2graph):
        x[0, :] = self.qa2node(qa_emb)
        node_extras = self.ntype_nscore_enc(torch.cat([node_types, node_scores], dim=-1))  # n_ntype + 1 --> D/2
        x = torch.cat([x, node_extras], dim=-1)  # 2D --> D
        x = self.act(x)

        edge_attr = self.etype_enc(edge_attr)

        x = self.gat_conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        p = global_mean_pool(x, batch=node2graph).squeeze()
        # p = self.act(p)

        # x = F.dropout(x, p=self.dropout, training=self.training)
        return x[0], p


# class CustomMessagePassing(MessagePassing):
#     def __init__(self, hid_dim, n_ntype, n_etype):
#         super(CustomMessagePassing, self).__init__(aggr='add')
#         self.hid_dim = hid_dim
#         self.n_ntype = n_ntype
#         self.n_etype = n_etype
#
#     def forward(self, x, node_types, node_scores, edge_index, edge_type):
#         return self.propagate(edge_index=edge_index, x=x)
