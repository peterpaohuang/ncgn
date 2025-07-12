
import os
import time
import copy
import torch
import argparse
import random
import numpy as np

from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.models import GCN, GAT

from src.dataset import STGraphDataset
from src.gnn import STAdaptiveGNN
from src.utils import ema

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
def convert_x(x, time, normalization):
    N = x.shape[0]
    x = x.reshape(N, 10, 10, 3)
    time = time.reshape(N, 10, 10, 1)

    for i in range(10):
        t = int(time[0, i, 0, 0] / .005)
        min_val = normalization[t, 0]
        max_val = normalization[t, 1]
        x[:, i, :, :] = (x[:, i, :, :] - min_val) / (max_val - min_val)

    return x.reshape(N, 100, 3)

def invert_x(x, time, normalization):
    N = x.shape[0]
    x = x.reshape(N, 10, 10, 3)
    time = time.reshape(N, 10, 10, 1)

    for i in range(10):
        t = int(time[0, i, 0, 0] / .005)
        min_val = normalization[t, 0]
        max_val = normalization[t, 1]

        x[:, i, :, :] = x[:, i, :, :] * (max_val - min_val) + min_val

    return x.reshape(N, 100, 3)

st_dataset_x = torch.from_numpy(np.load("data/st_train_x.npy"))
st_dataset_pos = torch.from_numpy(np.load("data/st_train_pos.npy"))
st_dataset_time = torch.from_numpy(np.load("data/st_train_time.npy"))

normalization = torch.from_numpy(np.load("data/st_normalization.npy"))

st_dataset_x = convert_x(st_dataset_x, st_dataset_time, normalization)

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

args = parser.parse_args()

print(f"GNN TYPE: {args.gnn_type}")
print(f"DMP SCHEDULE: {args.dmp_schedule}")
print(f"MODEL TYPE: {args.model_type}")


#########################################################################
# Constants
#########################################################################

torch.set_num_threads(8)

K_END = 5
train_dataset = STGraphDataset(
    st_dataset_x, st_dataset_pos, st_dataset_time, k_end=K_END,
    gnn_type=args.gnn_type, diffuser_type='flow_matching', 
    dmp_schedule=args.dmp_schedule

)

IN_DIM = 6
OUT_DIM = 3
BATCH_SIZE = 128
LR = 1e-3
WARMUP = 10
N_LAYERS = 3
EMA_DECAY = 0.95
WIDTH = 32
EPOCHS = 300

#########################################################################
# Setup Model
#########################################################################

model_func = {"GCN": GCN, "GAT": GAT}[args.model_type]
model = STAdaptiveGNN(
    model_func(
        WIDTH, WIDTH, N_LAYERS, WIDTH, norm=BatchNorm(WIDTH)
    ), 
    IN_DIM, WIDTH, OUT_DIM
).to('cuda')
model.train()
ema_model = copy.deepcopy(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def warmup_lr(step):
    return min(step, WARMUP) / WARMUP
sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

os.makedirs("weights/st", exist_ok=True)

if args.gnn_type == "dmp":
    save_dir = f"weights/st/flow_matching/{args.model_type}/{args.dmp_schedule}_{args.gnn_type}"
else:
    save_dir = f"weights/st/flow_matching/{args.model_type}/{args.gnn_type}"
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