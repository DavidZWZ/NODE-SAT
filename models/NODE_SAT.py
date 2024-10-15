import numpy as np
import torch
import torch.nn as nn
from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from torch.nn import MultiheadAttention
from torchdiffeq import odeint

class NODE_SAT(nn.Module):
    """
    Neural Ordinary Differential Equation with Self-Attention for Temporal Graph Learning
    """

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, input_dim: int, num_layers: int = 2, dropout: float = 0.1, device: str = 'cpu', time_param: float = 1, 
                 num_heads: int = 1):
        """
        Initialize the NODE_SAT model.
        """
        super(NODE_SAT, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        
        #input_dim is the number of neighbors
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.time_param = np.float32(time_param)    
        self.num_heads = num_heads
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=True)
        self.projection_layer = nn.Linear(self.edge_feat_dim + time_feat_dim, self.edge_feat_dim)

        self.ODE_SAT = nn.ModuleList([
            ODE_SAT(input_dim=self.input_dim, dropout=self.dropout, time_param=self.time_param, num_heads=self.num_heads)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.edge_feat_dim + self.node_feat_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20, time_gap: int = 2000):
        """
        Compute temporal embeddings for source and destination nodes.
        """
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)

        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         num_neighbors: int = 20, time_gap: int = 2000, time_param: float = 1):
        """
        Compute temporal embeddings for given nodes.
        """
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        nodes_neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        nodes_neighbor_time_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim + time_feat_dim)
        combined_features = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
        # Tensor, shape (batch_size, num_neighbors, num_channels)
        combined_features = self.projection_layer(combined_features)

        for ODE_SAT in self.ODE_SAT:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            combined_features = ODE_SAT(input_tensor=combined_features)

        # Tensor, shape (batch_size, num_channels)
        combined_features = torch.mean(combined_features, dim=1)

        # get temporal neighbors of nodes, including neighbor ids
        time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                                                          node_interact_times=node_interact_times,
                                                                                          num_neighbors=time_gap)

        # Tensor, shape (batch_size, time_gap, node_feat_dim)
        nodes_time_gap_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(time_gap_neighbor_node_ids)]

        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask = torch.from_numpy((time_gap_neighbor_node_ids > 0).astype(np.float32))
        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask[valid_time_gap_neighbor_node_ids_mask == 0] = torch.finfo(torch.float32).min
        # Tensor, shape (batch_size, time_gap)
        scores = torch.softmax(valid_time_gap_neighbor_node_ids_mask, dim=1).to(self.device)

        # Tensor, shape (batch_size, node_feat_dim), average over the time_gap neighbors
        nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)

        # Tensor, shape (batch_size, node_feat_dim), add features of nodes in node_ids
        output_node_features = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features[torch.from_numpy(node_ids)]

        # Tensor, shape (batch_size, node_feat_dim)
        node_embeddings = self.output_layer(torch.cat([combined_features, output_node_features], dim=1))

        return node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        Set the neighbor sampler for the model.
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

class ODEFunc(nn.Module):
    """
    ODE function for the Neural ODE.
    """

    def __init__(self, input_dim: int, dropout: float = 0.0):
        """
        Initialize the ODE function.
        """
        super(ODEFunc, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.ffn = nn.Linear(in_features=input_dim, out_features=input_dim)

    def forward(self, t, x: torch.Tensor, perturb=None):
        """
        Forward pass of the ODE function.
        """
        s=  self.ffn(x)
        return s
    
class ODEblock(nn.Module):
    """
    ODE block for solving the Neural ODE.
    """

    def __init__(self, odefunc, t = torch.tensor([0,1]), solver='euler'):
        """
        Initialize the ODE block.
        """
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc
        self.solver = solver

    def forward(self, x):
        """
        Forward pass of the ODE block.
        """
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method=self.solver)[1]
        return z
    
class attention_block(nn.Module):
    """
    Multi-head attention block.
    """

    def __init__(self, input_dim: int, dropout: float = 0.0, num_heads: int = 1):
        """
        Initialize the attention block.
        """
        super(attention_block, self).__init__()

        self.input_dim = input_dim
        self.att = MultiheadAttention(input_dim, num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the attention block.
        """
        return self.att(x,x,x)[0]


class ODE_SAT(nn.Module):
    """
    ODE with Self-Attention block.
    """

    def __init__(self, input_dim: int, dropout: float = 0.0, time_param: float = 1, num_heads: int = 1):
        """
        Initialize the ODE_SAT block.
        """
        super(ODE_SAT, self).__init__()
        self.time_param = time_param
        self.norm = nn.LayerNorm(input_dim)
        self.ode_block = ODEblock(ODEFunc(input_dim=input_dim, dropout=dropout), t= torch.tensor([0, self.time_param]))
        self.att = attention_block(input_dim=input_dim, dropout=dropout, num_heads=num_heads)

    def forward(self, input_tensor: torch.Tensor):
        """
        Forward pass of the ODE_SAT block.
        """
        hidden_tensor = self.norm(input_tensor.permute(0, 2, 1))
        hidden_tensor_1 = self.ode_block(hidden_tensor)
        hidden_tensor = self.att(hidden_tensor_1).permute(0, 2, 1)
        output_tensor = hidden_tensor + input_tensor

        return output_tensor