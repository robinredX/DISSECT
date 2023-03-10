import networkx as nx
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.utils as pyg_utils


def load_sample_names(path):
    with open(path, "r") as f:
        sample_list = f.readlines()
    sample_list = [x.strip() for x in sample_list][1::]
    return sample_list


def load_celltypes(path):
    with open(path, "r") as f:
        cell_type_list = f.readlines()
    cell_type_list = [x.strip() for x in cell_type_list][1::]
    return cell_type_list


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


def construct_networkx_graph(coords, dist_mat, adj_mat, X_real=None):
    graph = nx.from_numpy_array(adj_mat)
    # remove edge weights
    for edge in graph.edges:
        del graph.edges[edge]["weight"]
    nx.set_edge_attributes(
        graph, {(i, j): dist_mat[i, j] for i, j in graph.edges}, "edge_weight"
    )
    nx.set_edge_attributes(
        graph, {(i, j): np.array([dist_mat[i, j]]) for i, j in graph.edges}, "edge_attr"
    )
    nx.set_node_attributes(graph, {i: coords[i] for i in range(coords.shape[0])}, "pos")
    # also add real gene expression as node attributes
    if X_real is not None:
        nx.set_node_attributes(
            graph,
            {i: x for i, x in enumerate(X_real)},
            "x",
        )
    return graph


def sample_k_hop_subgraph(graph, k=5):
    num_nodes = graph.number_of_nodes()
    # select random node
    node = np.random.randint(num_nodes)
    # select k hop neighborhood
    subgraph = nx.ego_graph(graph, node, radius=k)
    return subgraph


def check_radius(raw_coords, radius=0.02, num_hops=1):
    coords = normalize_coords(raw_coords)
    graph = construct_networkx_graph(*construct_spatial_graph(coords, radius=radius))
    print(f"Radius: {radius}")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    subgraph = sample_k_hop_subgraph(graph, 1)
    nx.draw(subgraph, pos=nx.get_node_attributes(subgraph, "pos"), node_size=10)
    plt.show()


def split_graph(graph, num_nodes=64):
    # produce a list of graphs with num_nodes nodes
    num_nodes_orig = graph.num_nodes
    random_indices = np.random.permutation(num_nodes_orig)
    graph_list = []
    for i in range(num_nodes_orig // num_nodes):
        subgraph = graph[random_indices[i * num_nodes : (i + 1) * num_nodes]]
        graph_list.append(subgraph)
    return graph_list


def load_data(
    st_path="data/V1_Mouse_Brain_Sagittal_Anterior.h5ad",
    gene_expr_path="experiment/datasets",
):
    st_data = sc.read_h5ad(st_path)
    X_real = np.load(f"{gene_expr_path}/X_real_test.npy")
    X_real_train = np.load(f"{gene_expr_path}/X_real_train.npy")
    X_sim = np.load(f"{gene_expr_path}/X_sim.npy")
    y_sim = np.load(f"{gene_expr_path}/y_sim.npy")
    return st_data, X_real, X_real_train, X_sim, y_sim


def load_spatial_data(st_path="data/V1_Mouse_Brain_Sagittal_Anterior.h5ad"):
    st_data = sc.read_h5ad(st_path)
    return st_data


def load_prepared_data(gene_expr_path="experiment/datasets"):
    X_real = np.load(f"{gene_expr_path}/X_real_test.npy")
    X_real_train = np.load(f"{gene_expr_path}/X_real_train.npy")
    X_sim = np.load(f"{gene_expr_path}/X_sim.npy")
    y_sim = np.load(f"{gene_expr_path}/y_sim.npy")
    return X_real, X_real_train, X_sim, y_sim


def load_real_groundtruth(path=None, col_order=None):
    try:
        if col_order is None:
            y_real_df = pd.read_csv(path, sep="\t", index_col=0)
        else:
            y_real_df = pd.read_csv(path, sep="\t", index_col=0)[col_order]
        y_real = y_real_df.to_numpy()
        return y_real, y_real_df
    except Exception as e:
        print(e)
        return None


def compute_graph_list(graph, X_sim, y_sim, p=0.0, test=False):
    num_sims = X_sim.shape[0]
    num_nodes = graph.num_nodes

    # build real graph
    real_graph = graph

    if not test:
        # create fake coordinates to match dimensionality
        fake_pos = torch.zeros((num_nodes, 2)) - 1

        # build simulated graph
        # sample gene expression from simulated data
        sim_slice = np.random.choice(num_sims, size=num_nodes)
        x_sim = torch.Tensor(X_sim[sim_slice, :])
        y_sim_slice = torch.Tensor(y_sim[sim_slice, :])
        # define custom edge index instead of the spatial one
        # e.g. fully connected or no connections or random
        sim_edge_index = pyg_utils.erdos_renyi_graph(num_nodes, p, directed=False)
        # always add self loops
        loop_index = torch.arange(0, num_nodes, dtype=torch.long)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        sim_edge_index = torch.cat([sim_edge_index, loop_index], dim=1)
        sim_graph = Data(
            x=x_sim, edge_index=sim_edge_index, pos=fake_pos, y=y_sim_slice
        )
    else:
        sim_graph = Data()

    return [real_graph, sim_graph]


def prepare_graph_dataset(
    st_data,
    X_real,
    X_sim,
    y_sim,
    num_samples=100,
    num_hops=4,
    radius=0.02,
    p=0.5,
    sample_sugraphs=True,
    test=False,
    y_real=None,
):
    coords, dist_mat, adj_mat = construct_spatial_graph(
        st_data.obsm["spatial"].copy(), radius=radius
    )

    full_graph = construct_networkx_graph(coords, dist_mat, adj_mat, X_real)

    if sample_sugraphs:
        # sample subgraphs
        sub_graphs = [
            sample_k_hop_subgraph(full_graph, k=num_hops) for _ in range(num_samples)
        ]
        # convert to pytorch geometric graphs
        # slightly ineffient, should be improved
        sub_graphs = [pyg_utils.from_networkx(g) for g in sub_graphs]

        if test:
            raise NotImplementedError("Test mode not implemented for subgraphs")
        else:
            # produce data list
            data_list = [
                compute_graph_list(sub_graph, X_sim, y_sim, p=p)
                for sub_graph in sub_graphs
            ]
    else:
        # convert fullgraph to pytorch geometric graph and share edge index and node features
        full_graph = pyg_utils.from_networkx(full_graph)
        if y_real is not None:
            full_graph.y = torch.Tensor(y_real)
        data_list = [
            compute_graph_list(full_graph, X_sim, y_sim, p=p, test=test)
            for _ in range(num_samples)
        ]
    return data_list


def prepare_dataset(
    X_real_train,
    X_sim,
    y_sim,
    y_real=None,
    st_data=None,
):
    size = X_real_train.shape[0]
    # check whether coordinates available
    if st_data is not None:
        coords = normalize_coords(st_data.obsm["spatial"].copy())
        # repeat along axis 0
        coords = np.tile(coords, ((size // coords.shape[0]) + 1, 1))[0:size, :]
        coords = torch.Tensor(coords)

    X_real_train = torch.Tensor(X_real_train)
    X_sim = torch.Tensor(X_sim)
    y_sim = torch.Tensor(y_sim)
    if y_real is not None:
        y_real = np.tile(y_real, ((size // y_real.shape[0]) + 1, 1))[0:size, :]
        y_real = torch.Tensor(y_real)

    # convert this into one big datalist
    data_list = []
    for i in range(X_real_train.shape[0]):
        x_real = X_real_train[i, :][None, :]
        x_sim = X_sim[i, :][None, :]
        y_sim_sub = y_sim[i, :][None, :]

        # check whether ground truth available for real data
        if y_real is not None:
            y_real_sub = y_real[i, :][None, :]
        else:
            y_real_sub = None

        # check whether coordinates available
        if st_data is not None:
            pos = coords[i, :][None, :]
            fake_pos = torch.zeros((1, 2)) - 1
        else:
            pos = None
            fake_pos = None
        g_real = Data(
            x=x_real, y=y_real_sub, pos=pos, edge_index=None, edge_weight=None
        )
        g_sim = Data(x=x_sim, y=y_sim_sub, pos=fake_pos, edge_index=None)
        data_list.append([g_real, g_sim])
    return data_list


class VisiumDataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
