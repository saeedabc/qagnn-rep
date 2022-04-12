import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, global_mean_pool
from torch_geometric.data.batch import Batch
from torch_geometric.utils import add_self_loops, degree
from transformers import AutoModel, BertModel


class QAGNN(nn.Module):
    def __init__(self, lm_name, seq_len, cp_dim, hid_dim, n_ntype, n_etype, dropout):
        super(QAGNN, self).__init__()

        self.dropout = dropout

        self.text_enc = AutoModel.from_pretrained(lm_name, output_hidden_states=True)
        for param in self.text_enc.base_model.parameters():
            param.requires_grad = False

        lm_hid_dim = self.text_enc.config.hidden_size
        self.text_att = torch.nn.MultiheadAttention(embed_dim=lm_hid_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.qa2cp = nn.Sequential(
            nn.Linear(seq_len * lm_hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, cp_dim)
        )

        self.gnn = GNN(cp_dim, hid_dim, n_ntype, n_etype, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(cp_dim + 2 * hid_dim, 2 * hid_dim),
            nn.ReLU(),
            nn.Linear(2 * hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, 1)
            # nn.Sigmoid()
        )

    def forward(self, tbatch, gbatch):
        input_ids, input_masks, segment_ids, output_masks = tbatch
        text_all_hid_states = self.text_enc(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks)
        hid_states = text_all_hid_states[-1][-1]  # (tb, seq_len, lm_hid_dim)

        qa_emb, _ = self.text_att(query=hid_states, key=hid_states, value=hid_states, key_padding_mask=output_masks, need_weights=False)  # (tb, seq_len, lm_hid_dim)
        qa_emb = self.qa2cp(qa_emb.contiguous().view(qa_emb.size(0), -1))  # (tb, cp_emb)
        # qa_emb = Batch.from_data_list(data_list=list(qa_emb), follow_batch=gbatch.batch)  # ([gb,] lm_hid_dim,)
        # qa_emb = Batch(qa_emb, follow_batch=gbatch.batch)
        # print(qa_emb.size(), gbatch.x.size())
        qa_node_emb, pooled_graph_emb = self.gnn(qa_emb=qa_emb, x=gbatch.x, node_ids=gbatch.node_ids, node_types=gbatch.node_types, node_scores=gbatch.node_scores,
                        edge_index=gbatch.edge_index, edge_type=gbatch.edge_type, edge_attr=gbatch.edge_attr, node2graph=gbatch.batch)  # (hid_dim,), (hid_dim,)
        
        emb = torch.concat([qa_emb, qa_node_emb, pooled_graph_emb], dim=-1)

        emb = F.dropout(emb, p=self.dropout, training=self.training)
        emb = self.mlp(emb)

        return emb


class GNN(nn.Module):
    def __init__(self, cp_dim, hid_dim, n_ntype, n_etype, dropout):
        super(GNN, self).__init__()
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        # self.gelu

        self.ntype_nscore_enc = nn.Linear(n_ntype + 1, hid_dim // 2)

        self.x2h = nn.Linear(cp_dim + hid_dim // 2, hid_dim)
        # self.h2h = nn.Linear(2 * hid_dim, hid_dim)

        self.etype_enc = nn.Sequential(
            torch.nn.Linear(n_ntype + n_etype + n_ntype, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            self.act,
            nn.Linear(hid_dim, hid_dim),
            self.act
        )

        self.gat_conv = GATConv(hid_dim, hid_dim, add_self_loops=False, edge_dim=hid_dim)

    def forward(self, qa_emb, x, node_ids, node_types, node_scores, edge_index, edge_type, edge_attr, node2graph):
        def _working_graph(qa_emb, x):
            # print(qa_emb.size(), x.size())
            bs = qa_emb.size(0)
            n_nodes_pb = x.size(0) // bs
            for b_idx in range(bs):
                qa_node_idx = b_idx * n_nodes_pb
                x[qa_node_idx, :] = qa_emb[b_idx, :]
            return x

        def _extract_h0(h, bs):
            h0_selector = range(0, h.size(0), h.size(0) // bs)
            return h[h0_selector, :]

        assert qa_emb.size(-1) == x.size(-1)
        x = _working_graph(qa_emb=qa_emb, x=x)
        x_extras = self.ntype_nscore_enc(torch.cat([node_types, node_scores], dim=-1))  # n_ntype + 1 --> D/2
        x = torch.cat([x, x_extras], dim=-1)  # 2D --> D
        x = self.act(x)
        
        h = self.act(self.x2h(x))

        edge_attr = self.etype_enc(edge_attr)

        h = self.gat_conv(x=h, edge_index=edge_index, edge_attr=edge_attr)
        
        p = global_mean_pool(h, batch=node2graph)
        p = self.act(p)

        h0 = _extract_h0(h, bs=qa_emb.size(0))
        
        return h0, p


# class CustomMessagePassing(MessagePassing):
#     def __init__(self, hid_dim, n_ntype, n_etype):
#         super(CustomMessagePassing, self).__init__(aggr='add')
#         self.hid_dim = hid_dim
#         self.n_ntype = n_ntype
#         self.n_etype = n_etype
#
#     def forward(self, x, node_types, node_scores, edge_index, edge_type):
#         return self.propagate(edge_index=edge_index, x=x)
