import numpy as np
import torch
import torch.nn.functional
from torch_geometric.data import Data

from data_utils import load_input_tensors, load_sparse_adj_data_with_contextnode, MODEL_NAME_TO_CLASS


def make_one_hot(tensor, n_cats):
    return torch.nn.functional.one_hot(tensor, n_cats).float()


class QADataWrapper:
    def __init__(self, qids, labels, text_data, node_data, adj_data):                           # before:
        self.qids = qids                                                                        # n_qs
        self.labels = make_one_hot(labels, 5).view(-1)                                          # (n_qs,)
        self.text_data = [x.view(-1, x.size(2)) for x in text_data]                             # 4 * (n_qs, n_as, max_seq_len)
        self.node_data = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in node_data]      # 2 * (n_qs, n_as, N), (n_qs, n_as, N, 1), (n_qs, n_as)
        self.adj_data = [[t for qa in x for t in qa] for x in adj_data]                         # n_qs * 5 * (2, E), n_qs * 5 * (E, )

    def __len__(self):
        return self.labels.size(0)

    @property
    def edge_index(self): return self.adj_data[0]

    @property
    def edge_type(self): return self.adj_data[1]

    @property
    def text_input_ids(self): return self.text_data[0]

    @property
    def text_input_mask(self): return self.text_data[1]

    @property
    def text_segment_ids(self): return self.text_data[2]

    @property
    def text_output_mask(self): return self.text_data[3]

    @property
    def node_concept_ids(self): return self.node_data[0]

    @property
    def node_type_ids(self): return self.node_data[1]

    @property
    def node_scores(self): return self.node_data[2]

    @property
    def node_adj_lengths(self): return self.node_data[3]


class QAGNN_RawDataLoader:
    def __init__(self, cp_emb_path, train_stmt_path, train_adj_path, dev_stmt_path, dev_adj_path, test_stmt_path, test_adj_path,
                 batch_size, lm_name, n_ntype, n_etype, max_node_num, max_seq_length):

        self.batch_size = batch_size
        self.lm_name = lm_name
        self.model_type = MODEL_NAME_TO_CLASS[lm_name]
        self.max_node_num = max_node_num
        self.max_seq_length = max_seq_length
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        self.cp_emb = torch.tensor(np.load(cp_emb_path), dtype=torch.float)  # (799273, 1024)  # TODO
        self.cp_dim = self.cp_emb.size(1)
        # print(f'conceptnet embeddings shape: {cp_emb.size()}')

        self.train_stmt_path = train_stmt_path
        self.train_adj_path = train_adj_path
        self._train_qad = None

        self.dev_stmt_path = dev_stmt_path
        self.dev_adj_path = dev_adj_path
        self._dev_qad = None

        self.test_stmt_path = test_stmt_path
        self.test_adj_path = test_adj_path
        self._test_qad = None

    @property
    def train_qad(self):
        if self._train_qad is None:
            self._train_qad = self._load_qagnn_inputs(self.train_stmt_path, self.train_adj_path)
        return self._train_qad

    @property
    def dev_qad(self):
        if self._dev_qad is None:
            self._dev_qad = self._load_qagnn_inputs(self.dev_stmt_path, self.dev_adj_path)
        return self._dev_qad

    @property
    def test_qad(self):
        if self._test_qad is None:
            self._test_qad = self._load_qagnn_inputs(self.test_stmt_path, self.test_adj_path)
        return self._test_qad

    def _load_qagnn_inputs(self, stmt_path, adj_path):
        qids, labels, *encoder_data = load_input_tensors(stmt_path, self.model_type, self.lm_name, self.max_seq_length)
        num_choice = encoder_data[0].size(1)
        *decoder_data, adj_data = load_sparse_adj_data_with_contextnode(adj_path, self.max_node_num, num_choice)
        assert all(len(qids) == len(adj_data[0]) == x.size(0) for x in [labels] + encoder_data + decoder_data)

        # lim = 1000  # TODO
        # qids = qids[:lim]
        # labels = labels[:lim]
        # encoder_data = [x[:lim] for x in encoder_data]
        # decoder_data = [x[:lim] for x in decoder_data]
        # adj_data = [x[:lim] for x in adj_data]
        return QADataWrapper(qids, labels, encoder_data, decoder_data, adj_data)

    def train_dataset(self): return self._dataset(self.train_qad)

    def dev_dataset(self): return self._dataset(self.dev_qad)

    def test_dataset(self): return self._dataset(self.test_qad)

    def _dataset(self, qad):
        li = []
        for i in range(len(qad)):  # graph i
            node_ids = qad.node_concept_ids[i]  # (N, )
            node_embs = torch.zeros((node_ids.size(0), self.cp_dim))
            node_embs[1:, :] = self.cp_emb[node_ids[1:] - 1, :]  # (N, 1024) # TODO

            # node_embs = torch.normal(mean=0, std=0.5, size=(node_ids.size(0), 24))  # (N, d)
            node_types = make_one_hot(qad.node_type_ids[i], self.n_ntype)  # (N, n_ntype)
            node_scores = qad.node_scores[i]  # (N, 1)
            # x = torch.cat([node_embs, node_types, node_scores], dim=1)  # (N, d + n_ntype + 1)

            edge_index = qad.edge_index[i]  # (2, E)
            edge_type = torch.nn.functional.one_hot(qad.edge_type[i], self.n_etype)  # (E,)

            sids, tids = edge_index[0, :].squeeze(0), edge_index[1, :].squeeze(0)
            s_type, t_type = node_types[sids], node_types[tids]
            edge_attr = torch.cat([s_type, t_type, edge_type], dim=-1)  # (E, 2 * n_ntype + n_etype)

            y = qad.labels[i].float()

            d = Data(x=node_embs, node_ids=node_ids, node_types=node_types, node_scores=node_scores,
                     edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr, y=y)

            li.append(d)
        return li
