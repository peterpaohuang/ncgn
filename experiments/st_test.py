import math
import ot
import argparse
import os
import random
import torch
import numpy as np

from tqdm import trange

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *

from torchcfm.utils import *

from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
from torch_geometric.utils imdport unbatch
from torch_geometric.nn.models import GCN, GAT
from scipy.spatial.distance import cdist

from src.dataset import STGraphDataset
from src.gnn import STAdaptiveGNN


RANDOM_SEED = 0
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Additional settings for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

#########################################################################
# Setup Dataset
#########################################################################

time_points = 8
space_points = 12

def convert_x(x, time, normalization):
    N = x.shape[0]
    x = x.reshape(N, time_points, space_points, 3)
    time = time.reshape(N, time_points, space_points, 1)

    for i in range(time_points):
        t = int(time[0, i, 0, 0] / .005)
        min_val = normalization[t, 0]
        max_val = normalization[t, 1]
        x[:, i, :, :] = (x[:, i, :, :] - min_val) / (max_val - min_val)

    return x.reshape(N, time_points * space_points, 3)

def invert_x(x, time, normalization):
    N = x.shape[0]
    x = x.reshape(N, time_points, space_points, 3)
    time = time.reshape(N, time_points, space_points, 1)

    for i in range(10):
        t = int(time[0, i, 0, 0] / .005)
        min_val = normalization[t, 0]
        max_val = normalization[t, 1]
        x[:, i, :, :] = x[:, i, :, :] * (max_val - min_val) + min_val

    return x.reshape(N, time_points * space_points, 3)


#########################################################################
# Parse Args
#########################################################################
parser = argparse.ArgumentParser(description='Model Configuration Parser')

parser.add_argument(
    '--gnn_type',
    type=str,
    default='dmp',
    choices=['knn', 'fully_connected', 'dmp'],
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
parser.add_argument(
    '--task',
    type=str,
    default='unconditional',
    choices=['unconditional', 'forward_sim', 'interpolation', 'gene_imputation', 'space_imputation', 'knockout'],
    help='Prediction task'
)

args = parser.parse_args()

print(f"GNN TYPE: {args.gnn_type}")
print(f"DMP SCHEDULE: {args.dmp_schedule}")
print(f"MODEL TYPE: {args.model_type}")
print(f"PREDICTION TASK: {args.task}")

#########################################################################
# Constants
#########################################################################

torch.set_num_threads(16)

DEVICE='cuda'
IN_DIM = 6
WIDTH = 32
K_END = 5
N_LAYERS = 3
BATCH_SIZE = 10
OUT_DIM = 3
NFE = 200

#########################################################################
# Setup Model
#########################################################################

model_func = {"GAT": GAT, "GCN": GCN}[args.model_type]
model = STAdaptiveGNN(
    model_func(WIDTH, WIDTH, N_LAYERS, WIDTH, norm=BatchNorm(WIDTH)), IN_DIM, WIDTH, OUT_DIM
).to('cuda')

# Load the model
if args.gnn_type == "dmp":
    PATH = f"weights/st/flow_matching/{args.model_type}/{args.dmp_schedule}_{args.gnn_type}/299.pt"
else:
    PATH = f"weights/st/flow_matching/{args.model_type}/{args.gnn_type}/299.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=DEVICE, weights_only=True)
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

def calculate_dist(pred, gt):
    N = len(pred)
    pred = pred.reshape(N, -1)
    gt = gt.reshape(N, -1)
    pairwise_distances = cdist(pred, gt)
    weights = np.ones(N) / N
    
    emd = ot.emd2(weights, weights, pairwise_distances)
    return emd

normalization = torch.from_numpy(np.load("data/st_normalization.npy"))

###################       Unconditional          #############################
if args.task == "unconditional":
    st_dataset_x = torch.from_numpy(np.load("data/st_test_x.npy"))
    st_dataset_pos = torch.from_numpy(np.load("data/st_test_pos.npy"))
    st_dataset_time = torch.from_numpy(np.load("data/st_test_time.npy"))

    dataset_size = len(st_dataset_x)
    count = 0
    pred = torch.ones((dataset_size, time_points * space_points, 3))
    for batch_iter in trange(0, dataset_size, BATCH_SIZE):
        t_schedule = torch.linspace(0, 1, NFE, device='cuda')
        x_og = torch.randn(BATCH_SIZE, time_points * space_points, 3)

        step_size = 1 / NFE
        with torch.no_grad():
            st_dataset_pos_batch = st_dataset_pos[batch_iter: (batch_iter + BATCH_SIZE)]
            st_dataset_time_batch = st_dataset_time[batch_iter: (batch_iter + BATCH_SIZE)]

            for i, t in enumerate(t_schedule):
                # if x_og.is_cuda: x_og = x_og.detach().cpu()
                test_dataset = STGraphDataset(
                    x_og, st_dataset_pos_batch, st_dataset_time_batch, k_end=K_END,
                    t=torch.tensor([t]), test=True, gnn_type=args.gnn_type, 
                    dmp_schedule=args.dmp_schedule, diffuser_type='flow_matching'
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
                    # new_graph = new_graph.permute(1,0)
                    ode_f = new_graph
                    ode_f = ode_f.detach().cpu()

                    x_og = x_og + step_size * ode_f


        pred[batch_iter: (batch_iter + BATCH_SIZE)] = x_og
    st_dataset_x = convert_x(st_dataset_x, st_dataset_time, normalization)
    dist = calculate_dist(pred, st_dataset_x) / time_points
    print(f"Unconditional Wasserstein Dist: {dist}")

###################       FORWARD SIMULATIOn          #############################
elif args.task == "forward_sim":
    """
    Predict future trajectory based on n timepoints. Conditioned on first n timepoints
    """
    n = 1
    st_dataset_x = torch.from_numpy(np.load("data/st_test_x.npy")).reshape(-1, time_points, space_points, 3)
    st_dataset_pos = torch.from_numpy(np.load("data/st_test_pos.npy"))
    st_dataset_time = torch.from_numpy(np.load("data/st_test_time.npy"))

    dataset_size = len(st_dataset_x)
    count = 0
    pred = torch.ones((dataset_size, time_points, space_points, 3))
    for batch_iter in trange(0, dataset_size, BATCH_SIZE):
        t_schedule = torch.linspace(0, 1, NFE, device='cuda')
        x_og = torch.randn(BATCH_SIZE, time_points, space_points, 3)

        step_size = 1 / NFE
        with torch.no_grad():
            st_dataset_pos_batch = st_dataset_pos[batch_iter: (batch_iter + BATCH_SIZE)]
            st_dataset_time_batch = st_dataset_time[batch_iter: (batch_iter + BATCH_SIZE)]
            x_og[:, 0:n] = st_dataset_x[batch_iter: (batch_iter + BATCH_SIZE), 0:n]
            mask = torch.ones_like(x_og) # forward simulation mask
            mask[:, 0:n] = 0

            st_dataset_pos_batch = st_dataset_pos_batch.reshape(BATCH_SIZE, time_points * space_points, 1)
            st_dataset_time_batch = st_dataset_time_batch.reshape(BATCH_SIZE, time_points * space_points, 1)

            for i, t in enumerate(t_schedule):

                test_dataset = STGraphDataset(
                    x_og.reshape(BATCH_SIZE, time_points * space_points, 3), st_dataset_pos_batch, st_dataset_time_batch,
                    t=torch.tensor([t]), test=True, gnn_type=args.gnn_type, 
                    dmp_schedule=args.dmp_schedule, diffuser_type='flow_matching'
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
                    # new_graph = new_graph.permute(1,0)
                    ode_f = new_graph.reshape(BATCH_SIZE, time_points, space_points, 3)
                    ode_f = ode_f.detach().cpu()

                    x_og = x_og + mask * step_size * ode_f


        pred[batch_iter: (batch_iter + BATCH_SIZE)] = x_og

    st_dataset_x = convert_x(st_dataset_x, st_dataset_time, normalization)
    dist = calculate_dist(pred, st_dataset_x) / (time_points - n)

    print(f"Forward Pred Wasserstein Dist: {dist}")

###################       INTERPOLATION          #############################
elif args.task == "interpolation":
    """
    Interpolate time points. Conditioned on first n and last n timepoints
    """
    n = 1
    st_dataset_x = torch.from_numpy(np.load("data/st_test_x.npy")).reshape(-1, time_points, space_points, 3)
    st_dataset_pos = torch.from_numpy(np.load("data/st_test_pos.npy"))
    st_dataset_time = torch.from_numpy(np.load("data/st_test_time.npy"))

    dataset_size = len(st_dataset_x)
    count = 0
    pred = torch.ones((dataset_size, time_points, space_points, 3))
    for batch_iter in trange(0, dataset_size, BATCH_SIZE):
        t_schedule = torch.linspace(0, 1, NFE, device='cuda')
        x_og = torch.randn(BATCH_SIZE, time_points, space_points, 3)

        step_size = 1 / NFE
        with torch.no_grad():
            st_dataset_pos_batch = st_dataset_pos[batch_iter: (batch_iter + BATCH_SIZE)]
            st_dataset_time_batch = st_dataset_time[batch_iter: (batch_iter + BATCH_SIZE)]
            x_og[:, 0:n] = st_dataset_x[batch_iter: (batch_iter + BATCH_SIZE), 0:n]
            x_og[:, -n:] = st_dataset_x[batch_iter: (batch_iter + BATCH_SIZE), -n:]
            mask = torch.ones_like(x_og) # interpolation mask
            mask[:, :n] = 0
            mask[:, -n:] = 0

            st_dataset_pos_batch = st_dataset_pos_batch.reshape(BATCH_SIZE, time_points * space_points, 1)
            st_dataset_time_batch = st_dataset_time_batch.reshape(BATCH_SIZE, time_points * space_points, 1)

            for i, t in enumerate(t_schedule):              
                test_dataset = STGraphDataset(
                    x_og.reshape(BATCH_SIZE, time_points * space_points, 3), st_dataset_pos_batch, st_dataset_time_batch,
                    t=torch.tensor([t]), test=True, gnn_type=args.gnn_type, 
                    dmp_schedule=args.dmp_schedule, diffuser_type='flow_matching'
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
                    ode_f = new_graph.reshape(BATCH_SIZE, time_points, space_points, 3)
                    ode_f = ode_f.detach().cpu()

                    x_og = x_og + mask * step_size * ode_f


        pred[batch_iter: (batch_iter + BATCH_SIZE)] = x_og

    st_dataset_x = convert_x(st_dataset_x, st_dataset_time, normalization)
    dist = calculate_dist(pred, st_dataset_x) / (time_points - 2 * n)

    print(f"Interpolation Wasserstein Dist: {dist}")

###################       GENE IMPUTATION          #############################
elif args.task == "gene_imputation":
    """
    Impute missing genes. Conditioning on genes in all other spaces and timepoints
    """
    st_dataset_x = torch.from_numpy(np.load("data/st_test_x.npy")).reshape(-1, time_points, space_points, 3).float()
    st_dataset_pos = torch.from_numpy(np.load("data/st_test_pos.npy"))
    st_dataset_time = torch.from_numpy(np.load("data/st_test_time.npy"))

    dataset_size = len(st_dataset_x)
    count = 0
    pred = torch.ones((dataset_size, time_points, space_points, 3))
    conditioning_genes = [0,2] # thus, we are imputing gene 1
    for batch_iter in trange(0, dataset_size, BATCH_SIZE):
        t_schedule = torch.linspace(0, 1, NFE, device='cuda')
        x_og = torch.randn(BATCH_SIZE, time_points, space_points, 3)

        step_size = 1 / NFE
        with torch.no_grad():
            st_dataset_pos_batch = st_dataset_pos[batch_iter: (batch_iter + BATCH_SIZE)]
            st_dataset_time_batch = st_dataset_time[batch_iter: (batch_iter + BATCH_SIZE)]

            x_og[:, :, :, conditioning_genes] = st_dataset_x[batch_iter: (batch_iter + BATCH_SIZE), :, :, conditioning_genes]
            mask = torch.ones_like(x_og) # imputation mask
            mask[:, :, :, conditioning_genes] = 0
            
            st_dataset_pos_batch = st_dataset_pos_batch.reshape(BATCH_SIZE, time_points * space_points, 1)
            st_dataset_time_batch = st_dataset_time_batch.reshape(BATCH_SIZE, time_points * space_points, 1)

            for i, t in enumerate(t_schedule):
                
                test_dataset = STGraphDataset(
                    x_og.reshape(BATCH_SIZE, time_points * space_points, 3), st_dataset_pos_batch, st_dataset_time_batch,
                    t=torch.tensor([t]), test=True, gnn_type=args.gnn_type, 
                    dmp_schedule=args.dmp_schedule, diffuser_type='flow_matching'
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
                    ode_f = new_graph.reshape(BATCH_SIZE, time_points, space_points, 3)
                    ode_f = ode_f.detach().cpu()

                    x_og = x_og + mask * step_size * ode_f


        pred[batch_iter: (batch_iter + BATCH_SIZE)] = x_og

    st_dataset_x = convert_x(st_dataset_x, st_dataset_time, normalization)
    dist = calculate_dist(pred, st_dataset_x) / time_points

    print(f"Gene Imputation Wasserstein Dist: {dist}")

###################       SPACE IMPUTATION          #############################
elif args.task == "space_imputation":
    """
    Impute missing spaces. Conditioning on other spaces and timepoints
    """
    st_dataset_x = torch.from_numpy(np.load("data/st_test_x.npy")).reshape(-1, time_points, space_points, 3).float()
    st_dataset_pos = torch.from_numpy(np.load("data/st_test_pos.npy"))
    st_dataset_time = torch.from_numpy(np.load("data/st_test_time.npy"))

    dataset_size = len(st_dataset_x)
    count = 0
    pred = torch.ones((dataset_size, time_points, space_points, 3))
    conditioning_space_indices = [0,1,5,6,10,11]
    for batch_iter in trange(0, dataset_size, BATCH_SIZE):
        t_schedule = torch.linspace(0, 1, NFE, device='cuda')
        x_og = torch.randn(BATCH_SIZE, time_points, space_points, 3)

        step_size = 1 / NFE
        with torch.no_grad():
            st_dataset_pos_batch = st_dataset_pos[batch_iter: (batch_iter + BATCH_SIZE)]
            st_dataset_time_batch = st_dataset_time[batch_iter: (batch_iter + BATCH_SIZE)]
            x_og[:, :, conditioning_space_indices, :] = st_dataset_x[batch_iter: (batch_iter + BATCH_SIZE), :, conditioning_space_indices, :]
            mask = torch.ones_like(x_og) # imputation mask
            mask[:, :, conditioning_space_indices, :] = 0
            
            st_dataset_pos_batch = st_dataset_pos_batch.reshape(BATCH_SIZE, time_points * space_points, 1)
            st_dataset_time_batch = st_dataset_time_batch.reshape(BATCH_SIZE, time_points * space_points, 1)

            for i, t in enumerate(t_schedule):
                # if x_og.is_cuda: x_og = x_og.detach().cpu()
                
                test_dataset = STGraphDataset(
                    x_og.reshape(BATCH_SIZE, time_points * space_points, 3), st_dataset_pos_batch, st_dataset_time_batch,
                    t=torch.tensor([t]), test=True, gnn_type=args.gnn_type, 
                    dmp_schedule=args.dmp_schedule, diffuser_type='flow_matching'
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
                    ode_f = new_graph.reshape(BATCH_SIZE, time_points, space_points, 3)
                    ode_f = ode_f.detach().cpu()

                    x_og = x_og + mask * step_size * ode_f


        pred[batch_iter: (batch_iter + BATCH_SIZE)] = x_og

    st_dataset_x = convert_x(st_dataset_x, st_dataset_time, normalization)
    dist = calculate_dist(pred, st_dataset_x) / time_points

    print(f"Space Imputation Wasserstein Dist: {dist}")

###################       KNOCKOUT          #############################
elif args.task == "knockout":
    """
    Hard knockout specific gene, unconditional
    """
    st_dataset_x = torch.from_numpy(np.load("data/st_test_knockout_x.npy")).to(torch.float32).reshape(-1, time_points, space_points, 3)
    st_dataset_pos = torch.from_numpy(np.load("data/st_test_knockout_pos.npy")).to(torch.float32)
    st_dataset_time = torch.from_numpy(np.load("data/st_test_knockout_time.npy")).to(torch.float32)
    st_dataset_knockout = torch.from_numpy(np.load("data/st_test_knockout_values.npy")).to(torch.int)

    dataset_size = len(st_dataset_x)
    count = 0
    pred = torch.ones((dataset_size, time_points, space_points, 3))
    for batch_iter in trange(0, dataset_size, BATCH_SIZE):
        t_schedule = torch.linspace(0, 1, NFE, device='cuda')
        x_og = torch.randn(BATCH_SIZE, time_points, space_points, 3)
        ko_genes = st_dataset_knockout[batch_iter: (batch_iter + BATCH_SIZE)] # (batch_size,)

        step_size = 1 / NFE
        with torch.no_grad():
            st_dataset_pos_batch = st_dataset_pos[batch_iter: (batch_iter + BATCH_SIZE)]
            st_dataset_time_batch = st_dataset_time[batch_iter: (batch_iter + BATCH_SIZE)]
            for ko_i, ko_genen in enumerate(ko_genes): x_og[ko_i, :, :, ko_genen] = 0.0
            mask = torch.ones_like(x_og) # forward simulation mask

            for ko_i, ko_genen in enumerate(ko_genes): mask[ko_i, :, :, ko_genen] = 0

            st_dataset_pos_batch = st_dataset_pos_batch.reshape(BATCH_SIZE, time_points * space_points, 1)
            st_dataset_time_batch = st_dataset_time_batch.reshape(BATCH_SIZE, time_points * space_points, 1)

            for i, t in enumerate(t_schedule):

                test_dataset = STGraphDataset(
                    x_og.reshape(BATCH_SIZE, time_points * space_points, 3), st_dataset_pos_batch, st_dataset_time_batch,
                    t=torch.tensor([t]), test=True, gnn_type=args.gnn_type, 
                    dmp_schedule=args.dmp_schedule, diffuser_type='flow_matching'
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
                    ode_f = new_graph.reshape(BATCH_SIZE, time_points, space_points, 3)
                    ode_f = ode_f.detach().cpu()

                    x_og = x_og + mask * step_size * ode_f


        pred[batch_iter: (batch_iter + BATCH_SIZE)] = x_og

    st_dataset_x = convert_x(st_dataset_x, st_dataset_time, normalization)
    dist = calculate_dist(pred, st_dataset_x) / time_points

    print(f"Knockout Wasserstein Dist: {dist}")
else:
    raise f"Invalid testing task {args.task}"