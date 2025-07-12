import ot
import math
import argparse
import torch
import os
import random
import numpy as np
from tqdm import trange

from diffusers import DDPMScheduler

from src.diffusion import diffusion_sampling

import torch_geometric.transforms as T
from torch_geometric.nn.models import GCN, GAT
from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import ModelNet
from torch_geometric.utils import unbatch
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.data import Dataset, DataLoader

from scipy.spatial.distance import cdist

from src.dataset import ModelNetGraphDataset
from src.gnn import ModelNetAdaptiveGNN


#########################################################################
# Setup Dataset
#########################################################################

class PointSampler(object):
    """Custom transform to subsample points from ModelNet meshes."""
    def __init__(self, num_points):
        self.num_points = num_points
    
    def __call__(self, data):
        num_points = data.pos.shape[0]
        if num_points > self.num_points:
            # Random sampling without replacement
            idx = np.random.choice(num_points, self.num_points, replace=False)
            data.pos = data.pos[idx]
            if hasattr(data, 'normal'):
                data.normal = data.normal[idx]
            # if hasattr(data, 'x'):
            #     data.x = data.x[idx]
            data.filter_idx = True
        else:
            data.filter_idx = False

        return data

class NormalizeFurther(BaseTransform):
    def __init__(self):
        pass

    def forward(self, data):
        scale = (0.5 / data.pos.abs().max()) * 0.49999999
        data.pos = data.pos * scale

        return data

def load_modelnet(root, categories=None, num_points=400):
    """
    Load ModelNet dataset with point subsampling.
    
    Args:
        root (str): Path to store/load the dataset
        categories (list): List of category names to load. None loads all categories.
        num_points (int): Number of points to sample from each shape
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Define preprocessing transforms
    transform = T.Compose([
        PointSampler(num_points),
        T.NormalizeScale(),
        NormalizeFurther()
    ])
    
    # Load training set
    train_dataset = ModelNet(
        root=root,
        name="40",
        train=False,
        transform=transform,
    )

    return train_dataset

dataset = load_modelnet(
        root='data/ModelNet',  # Specify your desired path
        num_points=400
    )


class FilteredModelNet(Dataset):
    def __init__(self, datalist, min_points=400):
        super(FilteredModelNet, self).__init__()
        self.datalist = datalist
        
        # Filter the processed_file_names to only include valid samples
        valid_indices = []
        for idx in range(len(self.datalist)):
            data = self.datalist[idx]
            if data.pos.shape[0] >= min_points:
                valid_indices.append(idx)
        
        # Update the data list to only include valid samples
        self._datalist = [self.datalist[i] for i in valid_indices]
        
    def len(self):
        return len(self._datalist)
    
    def get(self, idx):
        return self._datalist[idx]

dataset = FilteredModelNet(dataset)[:2232]
dataset_size = len(dataset)

#########################################################################
# Parse Args
#########################################################################
parser = argparse.ArgumentParser(description='Model Configuration Parser')

parser.add_argument(
    '--gnn_type',
    type=str,
    default='dmp',
    choices=['knn', 'long_short', 'dmp'],
    help='Type of GNN mechanism to use'
)

parser.add_argument(
    '--dmp_schedule',
    type=str,
    default='exp',
    choices=['linear', 'relu', 'log', 'exp'],
    help='Schedule type for DMP'
)

parser.add_argument(
    '--model_type',
    type=str,
    default='GAT',
    choices=['GCN', 'GAT'],
    help='Type of model architecture'
)

args = parser.parse_args()

print(f"GNN TYPE: {args.gnn_type}")
print(f"DMP SCHEDULE: {args.dmp_schedule}")
print(f"MODEL TYPE: {args.model_type}")

#########################################################################
# Constants
#########################################################################
torch.set_num_threads(16)

K_END = 7
BATCH_SIZE = 8
DEVICE='cuda'
IN_DIM = 4
WIDTH = 64
N_LAYERS = 3
OUT_DIM = 3
NFE = 1000
DIFFUSER = DDPMScheduler(num_train_timesteps=NFE)

#########################################################################
# Setup Model
#########################################################################

model_func = {"GAT": GAT, "GCN": GCN}[args.model_type]
model = ModelNetAdaptiveGNN(
    model_func(WIDTH, WIDTH, N_LAYERS, WIDTH, norm=BatchNorm(WIDTH)), IN_DIM, WIDTH, OUT_DIM
).to('cuda')

# Load the model
if args.gnn_type == "dmp":
    PATH = f"weights/modelnet/diffusion/{args.model_type}/{args.dmp_schedule}_{args.gnn_type}/299.pt"
else:
    PATH = f"weights/modelnet/diffusion/{args.model_type}/{args.gnn_type}/299.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=DEVICE)
state_dict = checkpoint["ema_model"]
try:
    model.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
model.eval()


#########################################################################
# Evaluation
#########################################################################

count = 0

def calculate_dist(pred, gt):
    N = len(pred)
    pred = pred.reshape(N, -1)
    gt = gt.reshape(N, -1)
    pairwise_distances = cdist(pred, gt)
    weights = np.ones(N) / N
    
    emd = ot.emd2(weights, weights, pairwise_distances)
    return emd

pred = torch.ones((dataset_size, 400, 3))
gt = torch.cat([dataset[idx].pos for idx in range(dataset_size)], dim=0)

for batch_iter in trange(0, dataset_size, BATCH_SIZE):
    t_schedule = torch.linspace(0, 1, NFE, device='cuda')
    pos_og = torch.randn(BATCH_SIZE, 400, 3)

    step_size = 1 / NFE
    with torch.no_grad():
        t_schedule = torch.flip(t_schedule, dims=[0])

        for i, t in enumerate(t_schedule):
            x_og = pos_og
            test_dataset = ModelNetGraphDataset(
                x_og, pos_og, k_end=K_END,
                t=torch.tensor([t]), test=True, gnn_type=args.gnn_type, 
                dmp_schedule=args.dmp_schedule, diffuser_type='diffusion'
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                follow_batch=['x_cg', 'x_og']
            ) 

            for batch in test_loader:
                # since there is only one batch where the batch_size=n_samples, this will only loop once
                batch = batch.to(DEVICE)
                new_graph = model(batch)

                new_graph = torch.stack(unbatch(new_graph, batch.x_og_batch))
                diffusion_t = min(math.floor(NFE * t), NFE-1)
                prev_t = t_schedule[i + 1]  if i < len(t_schedule) - 1 else t_schedule[-1]
                diffusion_prev_t = min(math.floor(NFE * prev_t), NFE-1)
                pos_og = diffusion_sampling(DIFFUSER, new_graph.detach().cpu(), diffusion_t,diffusion_prev_t, pos_og)


    pred[batch_iter: (batch_iter + BATCH_SIZE)] = pos_og
    
dist = calculate_dist(pred, gt) / 400
print(f"Wasserstein Dist: {dist}")
