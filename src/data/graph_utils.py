import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
import torch_geometric.utils as pyg_utils
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt


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


def adj_to_edge_index(adj):
    """Convert adjacency matrix to edge index."""
    edge_index = adj.nonzero().t().contiguous()
    return edge_index


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
    subgraph = sample_k_hop_subgraph(graph, num_hops)
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