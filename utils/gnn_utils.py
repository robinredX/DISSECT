import networkx as nx
import scanpy as sc
import numpy as np
from scipy.spatial import distance_matrix
import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.utils as pyg_utils


def normalize_coords(coords):
    """Normalize coordinates to [0, 1] range."""
    coords = coords - coords.min(axis=0)
    coords = coords / coords.max()
    return np.float32(coords)


def construct_spatial_graph(coords, radius=0.02):
    """Construct a spatial graph from coordinates and radius."""
    coords = normalize_coords(coords)
    dist_mat = distance_matrix(coords, coords)
    # calculate adjacency matrix
    adj_mat = np.zeros_like(dist_mat)
    # set threshold such that each spot has maximum 6 neighbors for Visium data
    adj_mat[dist_mat < radius] = 1
    return np.float32(coords), np.float32(dist_mat), np.float32(adj_mat)


def construct_networkx_graph(coords, dist_mat, adj_mat, X_real):
    full_graph = nx.from_numpy_array(adj_mat)
    # remove edge weights
    for edge in full_graph.edges:
        del full_graph.edges[edge]["weight"]
    nx.set_edge_attributes(
        full_graph, {(i, j): dist_mat[i, j] for i, j in full_graph.edges}, "edge_weight"
    )
    nx.set_node_attributes(
        full_graph, {i: coords[i] for i in range(coords.shape[0])}, "pos"
    )
    # also add real gene expression as node attributes
    nx.set_node_attributes(
        full_graph,
        {i: x for i, x in enumerate(X_real)},
        "x",
    )
    return full_graph


def sample_k_hop_subgraph(graph, k=5):
    num_nodes = graph.number_of_nodes()
    # select random node
    node = np.random.randint(num_nodes)
    # select k hop neighborhood
    subgraph = nx.ego_graph(graph, node, radius=k)
    return subgraph


def load_data(
    st_path="data/V1_Mouse_Brain_Sagittal_Anterior.h5ad",
    gene_expr_path="experiment/datasets",
):
    st_data = sc.read_h5ad(st_path)
    X_sim = np.load(f"{gene_expr_path}/X_sim.npy")
    y_sim = np.load(f"{gene_expr_path}/y_sim.npy")
    X_real = np.load(f"{gene_expr_path}/X_real_test.npy")
    return st_data, X_real, X_sim, y_sim

def compute_graph_list(sub_graph, X_sim, y_sim, p=0.5):
    num_sims = X_sim.shape[0]
    num_nodes = sub_graph.number_of_nodes()
    # convert to pytorch geometric data
    sub_graph = pyg_utils.from_networkx(sub_graph)

    # build real graph
    real_graph = sub_graph
    x_real = real_graph.x.clone()
    # potentially concat pos and expression here
    real_graph.x = torch.concat([real_graph.x, real_graph.pos], dim=1)

    # create fake coordinates to match dimensionality
    fake_pos = torch.zeros((num_nodes, 2)) - 1

    # build simulated graph
    # sample gene expression from simulated data
    sim_slice = np.random.choice(num_sims, size=num_nodes)
    x_sim = X_sim[sim_slice, :]
    x_sim = torch.Tensor(x_sim)
    y_sim_slice = y_sim[sim_slice, :]
    y_sim_slice = torch.Tensor(y_sim_slice)
    # define custom edge index instead of the spatial one
    # e.g. fully connected or no connections or random
    sim_edge_index = pyg_utils.erdos_renyi_graph(num_nodes, p, directed=False)
    sim_graph = Data(x=x_sim, edge_index=sim_edge_index)
    sim_graph.x = torch.concat([sim_graph.x, fake_pos], dim=1)
    sim_graph.y = y_sim_slice

    # build mixture graph
    # generate mixture expression on the fly
    alpha = torch.Tensor(np.random.uniform(0, 1, size=(num_nodes, 1)))
    x_mix = (1 - alpha) * x_real + alpha * x_sim
    mix_graph = Data(x=x_mix, edge_index=sub_graph.edge_index)
    mix_graph.x = torch.concat([mix_graph.x, real_graph.pos], dim=1)

    return [real_graph, sim_graph, mix_graph]


def prepare_dataset(
    st_data,X_real, X_sim, y_sim, num_samples=100, num_hops=4, radius=0.02, p=0.5
):
    coords, dist_mat, adj_mat = construct_spatial_graph(
        st_data.obsm["spatial"].copy(), radius=radius
    )

    full_graph = construct_networkx_graph(coords, dist_mat, adj_mat, X_real)

    # sample subgraphs
    sub_graphs = [sample_k_hop_subgraph(full_graph, k=num_hops) for _ in range(num_samples)]

    # produce data list
    data_list = [compute_graph_list(sub_graph, X_sim, y_sim, p) for sub_graph in sub_graphs]

    return data_list


class VisiumDataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
