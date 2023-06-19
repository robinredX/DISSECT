from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data

from scipy.sparse import csr_matrix, coo_matrix

from src.data.graph_utils import (
    construct_spatial_graph,
    construct_networkx_graph,
    normalize_coords,
)
from src.data.utils import sparse_to_array


class DiSpatialGraphDataset(Dataset):
    """
    Dataset for multiple graphs.
    """

    def __init__(
        self,
        st_data_real,
        st_data_sim,
        y_real,
        y_sim,
        radius=0.02,
        test=False,
    ):
        self.st_data_real = st_data_real
        self.st_data_sim = st_data_sim
        self.radius = radius
        self.test = test
        self.graph_ids = F.one_hot(torch.arange(3)).float()

        self.X_real = sparse_to_array(st_data_real.X)
        self.g_real = construct_graph(
            st_data_real, self.X_real, y_real, radius, self.graph_ids[[0]]
        )

        if self.test:
            self.X_sim = None
            self.g_sim = Data()
        else:
            self.X_sim = sparse_to_array(st_data_sim.X)
            self.g_sim = construct_graph(
                st_data_sim, self.X_sim, y_sim, radius, self.graph_ids[[1]]
            )
            assert (
                self.X_real.shape[0] == y_real.shape[0]
            ), "X_real and y_real must have same number of spots"
            assert (
                self.X_real.shape[0] == self.X_sim.shape[0]
            ), "X_real and X_sim must have same number of spots"

        self.num_spots = self.X_real.shape[0]

    def __len__(self):
        # set number of samples to allow arbitrary batch sizes
        if self.test:
            return 1
        else:
            return 1024

    def __getitem__(self, idx):
        # always retrun the same pair of graphs for now
        # TODO: later extend this to sample a simulated graph
        return [self.g_real, self.g_sim]

    def move_to_device(self, device):
        self.g_real = self.g_real.to(device)
        self.g_sim = self.g_sim.to(device)


class SpatialDISSECTDataset(Dataset):
    """
    Spatial dataset base class.
    Randomly generates new simulated graphs for each sample.
    """

    def __init__(
        self,
        st_data,
        X_real,
        X_sim,
        y_sim,
        radius=0.02,
        p=0.0,
        test=False,
        y_real=None,
        num_samples=32,
    ):
        self.num_samples = num_samples
        self.num_spots = X_real.shape[0]
        self.num_simulations = X_sim.shape[0]
        self.radius = radius
        self.p = p
        self.test = test

        self.X_sim = torch.Tensor(X_sim)
        self.num_sims = self.X_sim.shape[0]
        self.y_sim = torch.Tensor(y_sim)

        self.graph_ids = F.one_hot(torch.arange(3)).float()

        self.real_graph = self.construct_real_graph(st_data, X_real, y_real)
        if not self.test:
            self.sim_base_graph = self.construct_sim_base_graph()

    def construct_real_graph(self, st_data, X_real, y_real):
        coords, dist_mat, adj_mat = construct_spatial_graph(
            st_data.obsm["spatial"].copy(), radius=self.radius
        )
        real_graph = construct_networkx_graph(coords, dist_mat, adj_mat, X_real)
        real_graph = pyg_utils.from_networkx(real_graph)
        if y_real is not None:
            real_graph.y = torch.Tensor(y_real)
        real_graph.id = self.graph_ids[[0]].expand(real_graph.num_nodes, -1)
        return real_graph

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # get new sim graph
        if self.test:
            sim_graph = Data()
        else:
            sim_graph = self.construct_new_sim_graph()
        return [self.real_graph, sim_graph]

    def construct_sim_base_graph(self):
        # create simulated base graph
        num_nodes = self.num_spots
        # create fake coordinates to match dimensionality
        fake_pos = torch.zeros((num_nodes, 2)) - 1

        # build simulated graph
        # define custom edge index instead of the spatial one
        # e.g. fully connected or no connections or random
        sim_edge_index = pyg_utils.erdos_renyi_graph(num_nodes, self.p, directed=False)
        # always add self loops
        loop_index = torch.arange(0, num_nodes, dtype=torch.long)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        sim_edge_index = torch.cat([sim_edge_index, loop_index], dim=1)
        sim_edge_attr = torch.zeros((sim_edge_index.shape[1], 1))
        sim_edge_weight = torch.zeros((sim_edge_index.shape[1],))
        sim_base_graph = Data(
            x=None,
            edge_index=sim_edge_index,
            edge_weight=sim_edge_weight,
            edge_attr=sim_edge_attr,
            pos=fake_pos,
            y=None,
        )
        sim_base_graph.id = self.graph_ids[[1]].expand(num_nodes, -1)
        return sim_base_graph

    def construct_new_sim_graph(self):
        # create new simulated graph by updating node features of the base graph
        sim_slice = np.random.choice(self.num_sims, size=self.num_spots, replace=False)
        x_sim = self.X_sim[sim_slice, :]
        y_sim_slice = self.y_sim[sim_slice, :]
        self.sim_base_graph.x = x_sim
        self.sim_base_graph.y = y_sim_slice
        return self.sim_base_graph

    def move_to_device(self, device):
        self.real_graph = self.real_graph.to(device)
        if not self.test:
            self.sim_base_graph = self.sim_base_graph.to(device)
        self.X_sim = self.X_sim.to(device)
        self.y_sim = self.y_sim.to(device)


def construct_graph(st_data, X, y=None, radius=0.02, graph_id=None):
    coords, dist_mat, adj_mat = construct_spatial_graph(
        st_data.obsm["spatial"].copy(), radius=radius
    )
    graph = construct_networkx_graph(coords, dist_mat, adj_mat, X)
    graph = pyg_utils.from_networkx(graph)
    if y is not None:
        graph.y = torch.Tensor(y)
    if graph_id is not None:
        graph.id = graph_id.expand(graph.num_nodes, -1)
    return graph


def get_spatial_train_and_test_set(
    st_data, X_real, X_sim, y_sim, radius, p=0.0, y_real=None, num_samples=32
):
    train_data = SpatialDISSECTDataset(
        st_data,
        X_real,
        X_sim,
        y_sim,
        radius=radius,
        p=p,
        num_samples=num_samples,
    )

    test_data = SpatialDISSECTDataset(
        st_data,
        X_real,
        X_sim,
        y_sim,
        num_samples=1,
        radius=radius,
        test=True,
        y_real=y_real,
    )
    return train_data, test_data


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
