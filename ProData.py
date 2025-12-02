import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from typing import Any, Optional

from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_dense_adj)

Feature_Path = "./Feature/"
MAP_CUTOFF = 14
ATOM_MAP_CUTOFF = 4
DIST_NORM = 15
NUM_VEC = 10

def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:

    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

def compute_lapse(data):

    assert data.edge_index is not None
    num_nodes = data.num_nodes
    assert num_nodes is not None
    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        data.edge_weight,
        normalization='sym',
        num_nodes=num_nodes,
    )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    is_undirected = True
    if num_nodes < 100:
        from numpy.linalg import eig, eigh
        eig_fn = eig if not is_undirected else eigh

        eig_vals, eig_vecs = eig_fn(L.todense())  # type: ignore
    else:
        from scipy.sparse.linalg import eigs, eigsh
        eig_fn = eigs if not is_undirected else eigsh

        eig_vals, eig_vecs = eig_fn(  # type: ignore
            L,
            k= NUM_VEC + 1,
            which='SR' if not is_undirected else 'SA',
            return_eigenvectors=True,
        )
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])

    se = torch.from_numpy(eig_vecs[:, 1:NUM_VEC + 1])
    sign = -1 + 2 * torch.randint(0, 2, (NUM_VEC,))
    se *= sign

    data = add_node_attr(data, se, attr_name='se')
    return data

def add_zeros(data):
    z = data.edge_index.new_zeros(data.edge_index.shape[1])
    data.edge_attr = z
    return data

def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)
def embedding(sequence_name):
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)

def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)

def cal_edges(sequence_name, radius=MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(int)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list, dist_matrix


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix

def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = Batch.from_data_list(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix

class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, psepos_path='./Feature/psepos/Train335_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist


    def __getitem__(self, index):
        sequence_name = self.names[index]

        label = np.array(self.labels[index])

        pos = self.residue_psepos[sequence_name]

        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos

        pos = torch.from_numpy(pos)


        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)
        node_features = torch.from_numpy(node_features)

        res_atom_features = get_res_atom_features(sequence_name)
        res_atom_features = torch.from_numpy(res_atom_features)
        node_features = torch.cat([node_features, res_atom_features], dim=-1)

        node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        radius_index_list, dismap = cal_edges(sequence_name, MAP_CUTOFF)

        edge_feat = self.cal_edge_attr(radius_index_list, pos)
        edge_index_tensor = torch.tensor(radius_index_list)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)
        edge_feat = torch.tensor(edge_feat)
        label_tensor = torch.tensor(label)

        pos = pos.float()

        G = Data(x=node_features, edge_index=edge_index_tensor, y=label_tensor, pos=pos, edge_attr=edge_feat)
        G = add_zeros(G)
        G = compute_lapse(G)

        return G



    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)
        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()

        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()


        radius_attr_list = np.array([distance, cos])
        return radius_attr_list





