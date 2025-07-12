import os
import time
import copy
import argparse
import torch
import random
import numpy as np

from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.models import GCN, GAT
from torch_geometric.datasets import ModelNet

from src.dataset import ModelNetGraphDataset
from src.gnn import ModelNetAdaptiveGNN
from src.utils import ema

#########################################################################
# Setup Dataset
#########################################################################

class PointSampler(object):
    """Custom transform to subsample points from ShapeNet meshes."""
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
        train=True,
        transform=transform
    )
    
    return train_dataset

dataset = load_modelnet(
        root='./data/ModelNet',  # Specify your desired path
        num_points=400
    )

print(f"Number of training samples: {len(dataset)}")

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
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

args = parser.parse_args()

print(f"GNN TYPE: {args.gnn_type}")
print(f"DMP SCHEDULE: {args.dmp_schedule}")
print(f"MODEL TYPE: {args.model_type}")

#########################################################################
# Constants
#########################################################################
torch.set_num_threads(8)

K_END = 7
train_dataset = ModelNetGraphDataset(
    dataset, dataset, k_end=K_END, 
    gnn_type=args.gnn_type, diffuser_type='diffusion', dmp_schedule=args.dmp_schedule
)

IN_DIM = 4
OUT_DIM = 3
BATCH_SIZE = 128
LR = 1e-4
WARMUP = 10
N_LAYERS = 3
EMA_DECAY = 0.95
WIDTH = 64
EPOCHS = 300

#########################################################################
# Setup Model
#########################################################################

model_func = {"GCN": GCN, "GAT": GAT}[args.model_type]
model = ModelNetAdaptiveGNN(
    model_func(
        WIDTH, WIDTH, N_LAYERS, WIDTH, norm=BatchNorm(WIDTH),
    ), IN_DIM, WIDTH, OUT_DIM
).to('cuda')
model.train()
ema_model = copy.deepcopy(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def warmup_lr(step):
    return min(step, WARMUP) / WARMUP
sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

if args.gnn_type == "dmp":
    save_dir = f"weights/modelnet/diffusion/{args.model_type}/{args.dmp_schedule}_{args.gnn_type}"
else:
    save_dir = f"weights/modelnet/diffusion/{args.model_type}/{args.gnn_type}"
os.makedirs(save_dir, exist_ok=True)

# show model size
model_size = 0
for param in model.parameters():
    model_size += param.data.nelement()
print("Model params: %.2f M" % (model_size / 1024 / 1024))

#########################################################################
# Training
#########################################################################

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    pin_memory=True, 
    num_workers=2,
    follow_batch=['x_og', 'x_cg'],
)
batch = next(iter(train_loader))

start = time.time()
for k in tqdm(range(EPOCHS)):
    total_loss = 0
    for i, batch in enumerate(train_loader):
        batch = batch.to('cuda')
        optimizer.zero_grad()

        vt = model(batch)

        loss = torch.mean((vt - batch.target) ** 2)
        loss.backward()

        optimizer.step()
        sched.step()

        ema(model, ema_model, EMA_DECAY)

        total_loss += loss.item()

    if (k + 1) % 5 == 0:
        print(f"Loss at epoch {k}: {total_loss}")
        torch.save(
            {
                "net_model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "sched": sched.state_dict(),
                "optim": optimizer.state_dict(),
                "step": k,
            },
            f"{save_dir}/{k}.pt",
        )    
