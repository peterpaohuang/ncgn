import torch

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

from diffusers import DDPMScheduler

from torch_geometric.nn import knn_graph
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Dataset

from src.schedules import exp_schedule, log_schedule, linear_schedule, relu_schedule
from src.utils import _avg_pool_x, get_vector_edge_attr, sinusoidal_embedding, voxel_clustering

def sample_long_range_edges(pos, k=5):
    """
    Uniform sampling random graph
    """
    # Get dense adjacency matrix
    adj_matrix = torch.ones(len(pos), len(pos))
    adj_matrix.fill_diagonal_(0)

    # Invert distances for sampling probabilities (add small epsilon to avoid division by zero)
    inv_dist = torch.ones((len(pos), len(pos))) # inverse cubic distance
    
    # Set diagonal to 0 to avoid self-loops
    inv_dist.fill_diagonal_(0)
    
    probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
    
    # Sample edges for all nodes at once
    sampled = torch.multinomial(probs, k, replacement=False)
    
    # Create edge index from sampled edges
    row = torch.arange(adj_matrix.size(1)).repeat_interleave(k)
    col = sampled.flatten()
    new_edges = torch.stack([row, col])
    
    return new_edges

class AdaptiveGraph(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_og':
            return self.x_og.size(0)
        
        if key == 'edge_index_cg':
            return self.x_cg.size(0)
        
        if key == 'og_to_cg_edge_index':
            return torch.tensor([[self.x_og.size(0)], [self.x_cg.size(0)]])
        
        return super().__inc__(key, value, *args, **kwargs)

class GraphDataset(Dataset):
    def __init__(self, k=5, time_emb_dim=1,
            cluster_inducing_method="voxel", k_start=200, k_end=5,
            t=None, test=False, diffuser_type="flow_matching", gnn_type=None,
            dmp_schedule="exp", **kwargs):
        super(GraphDataset, self).__init__(**kwargs)
        self.k = k
        self.time_emb_dim = time_emb_dim
        self.cluster_inducing_methods = cluster_inducing_method
        self.k_start = k_start
        self.k_end = k_end
        self.diffuser_type = diffuser_type # flow_matching, diffusion
        if self.diffuser_type == "flow_matching":
            self.diffuser = ConditionalFlowMatcher(sigma=0.0)
        elif self.diffuser_type == "diffusion":
           self.diffusion_timesteps = 1000
           self.diffuser = DDPMScheduler(num_train_timesteps=self.diffusion_timesteps) # 1000 timesteps
        else:
            raise "Invalid diffusion type"
        
        self.cluster_inducing_method = cluster_inducing_method

        self.gnn_type = gnn_type # knn, long_short, dmp
        self.test = test
        self.dmp_schedule = {
            "exp": exp_schedule,
            "log": log_schedule,
            "linear": linear_schedule,
            "relu": relu_schedule,
        }[dmp_schedule]

        self.t = t

    def len(self):
        pass

    def get(self, idx):
        pass

class STGraphDataset(GraphDataset):
    def __init__(
            self, features, positions, times, cluster_inducing_method="voxel", **kwargs
        ):
        super(STGraphDataset, self).__init__(cluster_inducing_method=cluster_inducing_method, **kwargs)
        self.features = features
        self.positions = positions
        self.times = times

    def len(self):
        return len(self.features)

    def get(self, idx):
        features = self.features[idx]
        pos = torch.cat(
            [
                self.positions[idx] / 54.0,
                self.times[idx] / 90.0 
                # makes position:time 2:1 for clustering
            ],
            dim=1
        )

        x1 = features
        
        pos_og = pos.float()

        x0 = torch.randn_like(x1)
        
        if not self.test:
          if self.diffuser_type == "flow_matching":
              t, x_og, target = self.diffuser.sample_location_and_conditional_flow(x0.unsqueeze(0), x1.unsqueeze(0))
          elif self.diffuser_type == "diffusion":
              target = x0
              t = torch.rand(1)
              diffusion_t = (t * self.diffusion_timesteps).to(torch.int)
              x_og = self.diffuser.add_noise(x1, x0, diffusion_t)
          else:
              raise "Invalid diffuser type"
        else:
          t = self.t
          x_og = features

        x_og = x_og.squeeze(0)
        x_og = torch.cat([x_og, pos_og], dim=-1)
        if not self.test:
           target = target.squeeze(0)
        else:
           target = None

        n_nodes_og = len(x_og)

        if self.gnn_type == "knn":
          n_nodes_cg = n_nodes_og
        elif self.gnn_type == "fully_connected":
          n_nodes_cg = n_nodes_og
        elif self.gnn_type == "dmp":
          n_nodes_cg, n_edges = self.dmp_schedule(t, self.k_end, n_nodes_og)
        else:
           raise "Invalid GNN type"

        if n_nodes_cg == n_nodes_og:
            # in very low noise levels, we don't coarse grain at all
            cluster_assignment = torch.arange(n_nodes_og).to(x_og.device)
        else:
            if self.cluster_inducing_method == "kmeans":
                # create n_clusters where cluster feature is NNConv of all nodes in cluster
                from torch_kmeans import KMeans
                cluster_assignment = KMeans(
                    verbose=False, n_clusters=n_nodes_cg
                ).fit_predict(pos_og.unsqueeze(0)).squeeze()
            elif self.cluster_inducing_method == "voxel":
                cluster_assignment, n_nodes_cg = voxel_clustering(pos_og, n_nodes_cg)
            else:
                raise
            
        # cluster
        cluster, perm = consecutive_cluster(cluster_assignment)
        n_nodes_cg = cluster.max() + 1
        x_cg = _avg_pool_x(cluster, x_og)
        pos_cg = pool_pos(cluster, pos_og)

        if self.gnn_type == "knn":
            edge_index_cg = knn_graph(pos_cg, k=self.k_end)
        elif self.gnn_type == "fully_connected":
            adj_matrix = torch.ones(n_nodes_og, n_nodes_og)
            adj_matrix.fill_diagonal_(0)
            edge_index_cg = dense_to_sparse(adj_matrix)[0]
        elif self.gnn_type == "dmp":
            edge_index_cg = knn_graph(pos_cg, k=n_edges)
        else:
           raise "Invalid GNN type"

        og_to_cg_edge_index = torch.tensor([[i for i in range(n_nodes_og)], cluster.tolist()])
        # fill out edge weights as inverse distances between pos
        # edge_attr_og = get_edge_attr(edge_index, pos)
        edge_attr_cg = get_vector_edge_attr(edge_index_cg, pos_cg)
        og_to_cg_edge_attr = get_vector_edge_attr(og_to_cg_edge_index, pos_og, pos_other=pos_cg)
        
        t_emb_og = sinusoidal_embedding(t, self.time_emb_dim).repeat_interleave(n_nodes_og, 1).permute(1, 0)
        t_emb_cg = sinusoidal_embedding(t, self.time_emb_dim).repeat_interleave(n_nodes_cg, 1).permute(1, 0)

        # concatenate time and position
        x_og = torch.cat([x_og, t_emb_og], dim=-1).float() # concat noise t embeddings with data
        x_cg = torch.cat([x_cg, t_emb_cg], dim=-1).float() # concat noise t embeddings with data

        edge_t_emb_cg = sinusoidal_embedding(t, self.time_emb_dim).repeat_interleave(len(edge_attr_cg), 1).permute(1, 0)
        edge_attr_cg =  torch.cat([edge_attr_cg, edge_t_emb_cg], dim=-1).float()
        og_to_cg_edge_attr =  torch.cat([og_to_cg_edge_attr, t_emb_og], dim=-1).float()

        adaptive_graph = AdaptiveGraph(
            x_cg=x_cg, pos_cg=pos_cg, edge_index_cg=edge_index_cg, edge_attr_cg=edge_attr_cg, 
            x_og=x_og, pos_og=pos_og,
            og_to_cg_edge_index=og_to_cg_edge_index, og_to_cg_edge_attr=og_to_cg_edge_attr, 
            target=target,
        )

        return adaptive_graph
    
class ModelNetGraphDataset(GraphDataset):
    def __init__(self, features, positions, **kwargs):
        super(ModelNetGraphDataset, self).__init__(**kwargs)
        self.features = features
        self.positions = positions

    def len(self):
        return len(self.positions)

    def get(self, idx):
        if isinstance(self.positions, torch.Tensor):
            pos1 = self.positions[idx]
        else: 
            pos1 = self.positions[idx].pos

        pos0 = torch.randn_like(pos1)

        if not self.test:
          if self.diffuser_type == "flow_matching":
              t, pos_og, target = self.diffuser.sample_location_and_conditional_flow(pos0.unsqueeze(0), pos1.unsqueeze(0))
          elif self.diffuser_type == "diffusion":
              target = pos0
              t = torch.rand(1)
              diffusion_t = (t * self.diffusion_timesteps).to(torch.int)
              pos_og = self.diffuser.add_noise(pos1, pos0, diffusion_t)
          else:
              raise "Invalid diffuser type"
        else:
          t = self.t
          pos_og = pos1

        pos_og = pos_og.squeeze(0)

        if not self.test:
           target = target.squeeze(0)
        else:
           target = None

        n_nodes_og = len(pos_og)
        if self.gnn_type == "knn":
          n_nodes_cg = n_nodes_og
        elif self.gnn_type == "long_short":
          n_nodes_cg = n_nodes_og
        elif self.gnn_type == "dmp":
          n_nodes_cg, n_edges = self.dmp_schedule(t, self.k_end, n_nodes_og)
        else:
           raise "Invalid GNN type"
        
        if n_nodes_cg == n_nodes_og:
            # in very low noise levels, we don't coarse grain at all
            cluster_assignment = torch.arange(n_nodes_og).to(pos_og.device)
        else:
            cluster_assignment, n_nodes_cg = voxel_clustering(pos_og, n_nodes_cg) 

        # cluster
        cluster, perm = consecutive_cluster(cluster_assignment)
        pos_cg = _avg_pool_x(cluster, pos_og)

        if self.gnn_type == "knn":
            edge_index_cg = knn_graph(pos_cg, k=self.k_end)
        elif self.gnn_type == "long_short":
            edge_index_cg = sample_long_range_edges(pos_cg, k=self.k_end) # long range
        elif self.gnn_type == "dmp":
            edge_index_cg = knn_graph(pos_cg, k=n_edges)
        else:
           raise "Invalid GNN type"
        
        og_to_cg_edge_index = torch.tensor([[i for i in range(n_nodes_og)], cluster_assignment.tolist()])
        # fill out edge weights as inverse distances between pos
        # edge_attr_og = get_edge_attr(edge_index, pos)
        edge_attr_cg = get_vector_edge_attr(edge_index_cg, pos_cg)
        og_to_cg_edge_attr = get_vector_edge_attr(og_to_cg_edge_index, pos_og, pos_other=pos_cg)
        
        t_emb_og = sinusoidal_embedding(t, self.time_emb_dim).repeat_interleave(n_nodes_og, 1).permute(1, 0)
        t_emb_cg = sinusoidal_embedding(t, self.time_emb_dim).repeat_interleave(n_nodes_cg, 1).permute(1, 0)

        # concatenate time and position
        x_og = torch.cat([pos_og, t_emb_og], dim=-1).float() # concat noise t embeddings with data
        x_cg = torch.cat([pos_cg, t_emb_cg], dim=-1).float() # concat noise t embeddings with data

        edge_t_emb_cg = sinusoidal_embedding(t, self.time_emb_dim).repeat_interleave(len(edge_attr_cg), 1).permute(1, 0)
        edge_attr_cg =  torch.cat([edge_attr_cg, edge_t_emb_cg], dim=-1).float()
        og_to_cg_edge_attr =  torch.cat([og_to_cg_edge_attr, t_emb_og], dim=-1).float()

        adaptive_graph = AdaptiveGraph(
            x_cg=x_cg, pos_cg=pos_cg, edge_index_cg=edge_index_cg, edge_attr_cg=edge_attr_cg, 
            x_og=x_og, pos_og=pos_og,
            og_to_cg_edge_index=og_to_cg_edge_index, og_to_cg_edge_attr=og_to_cg_edge_attr, 
            target=target,
        )

        return adaptive_graph