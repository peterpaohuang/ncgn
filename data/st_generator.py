# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from copy import deepcopy
# from scipy.io import savemat
# import pdb
# import numpy as np
from tqdm import trange
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy
from scipy.io import savemat

# The laplacian of values X from function X(s) over physical space S
# NOTE: The output size is the input size - 2
def laplacian_1d(X,ds):

    assert X.ndim==1, "The fucntion is only for the laplacian over one-dimensional space (i.e. line)"

    X_left = X[0:-2] # X(s-ds)
    X_right = X[2:] # X(s+ds)
    X_center = X[1:-1] # X(s)

    return (X_left+X_right-2*X_center)/ds**2

# Calculates the time velocity for variables at fixed time for X of size (space, variables)
# NOTE: the output space size the input space size - 2
def Velocity(X, ds):

    assert X.ndim == 2, "The dimension of the input must be two: (space, variables). Time is fixed."

    a_sox9 = 0
    a_bmp = 0
    a_wnt = 0
    k2 = 1
    k3 = -1
    k4 = 1.27
    k5 = -0.1
    k7 = 1.59
    k9 = -0.1
    d_b = 1
    d_w = 2.5

    Vsox9 = a_sox9 + k2*X[1:-1,1] + k3*X[1:-1,2] - X[1:-1,0]**3
    Vbmp  = a_bmp  + k4*X[1:-1,0] + k5*X[1:-1,1] + d_b*laplacian_1d(X[:,1], ds)
    Vwnt  = a_wnt  + k7*X[1:-1,0] + k9*X[1:-1,2] + d_w*laplacian_1d(X[:,2], ds)

    Vsox9 = np.expand_dims(Vsox9, axis=1)
    Vbmp = np.expand_dims(Vbmp, axis=1)
    Vwnt = np.expand_dims(Vwnt, axis=1)

    return np.concatenate([Vsox9,Vbmp,Vwnt],axis=1)

import pdb


#############################################################
# Unconditional Generation Train and Test Set
#############################################################

Num_of_variables = 3
ds = .2
dt = .005 # dt<ds**2 to keep the equations stable
T_max = 100
train_set = 500
sampling_interval = dt
S = np.arange( 0, 60, ds) # Space range
T = np.arange( 0, T_max + dt, dt) # Time range
num_nodes = 100
S_size = len(S)
T_size = len(T) - 1
dataset_size = int(S_size * T_size / num_nodes)

train_time_points = 10
train_space_points = 10

normalization = np.nan * np.ones([T_size, 2]) # 0 = min, 1 = max
dataset = np.nan * np.ones([train_set, train_time_points, train_space_points, Num_of_variables])
time_indices = np.arange(0, T_size, int(T_size / train_time_points), dtype=int)
space_indices = np.arange(0, S_size, int(S_size / train_space_points), dtype=int)

S_dataset = np.nan * np.ones([train_set, train_time_points, train_space_points, 1])
T_dataset = np.nan * np.ones([train_set, train_time_points, train_space_points, 1])

for test_i in trange(train_set):
    X = np.random.uniform(-0.01, 0.01, [ len(S), Num_of_variables] )
    X_plot = np.nan * np.ones( [ T_max, len(S), Num_of_variables] )
    X_store = np.nan * np.ones( [ int(1/sampling_interval)*T_max, len(S), Num_of_variables] )

    
    for i in range(1,len(T)):
        # print(i)
        X[1:-1] = X[1:-1] + dt * Velocity(X, ds)
        # Neumann conditions: The derivatives at the edges are null
        X[0] = X[1]
        X[-1] = X[-2]

        if (i%int(1/dt)==0):
            # print(T[i])
            X_plot[int(i/int(1/dt)-1)] = deepcopy(X)

        if (i%int(1/dt*sampling_interval)==0):
            X_store[int(i/int(1/dt*sampling_interval)-1)] = deepcopy(X)

        if np.isnan(normalization[i - 1]).any():
            normalization[i - 1] = [X.min(), X.max()]
        else:
            normalization[i - 1] = [min(normalization[i - 1][0], X.min()), max(normalization[i - 1][0], X.max())]
    
    dataset[test_i] = X_store[time_indices][:, space_indices]
    
    S_dataset[test_i] = np.repeat(S[space_indices][np.newaxis, :], train_time_points, axis=0)[..., np.newaxis]
    T_dataset[test_i] = np.dstack([T[time_indices]] * train_space_points)[..., np.newaxis]

# fig, ax = plt.subplots()
# def update(i):
#     plt.cla()
#     ax.set_ylim(-10, 10)
#     ax.set_title("Time = "+str(i))
#     ax.set_xlabel("Space")
#     ax.set_ylabel("Concentration")
#     ax.set_yticklabels("")
#     plt.plot(S_dataset[0][0], dataset[0, i ,: ,0], label="Sox9", color='#ED1E24', lw=3)
#     plt.plot(S_dataset[0][0], dataset[0, i ,: ,1], label="BMP", color='#0EAA4B', lw=3)
#     plt.plot(S_dataset[0][0], dataset[0, i ,: ,2], label="WNT", color='#3853A5', lw=3)
#     # plt.legend(loc=1)

# update(99)

# plt.savefig('time_99.png')

# # np.save('data/digit_train_x.npy', dataset.reshape((len(dataset), train_time_points * train_space_points, 3)))
# np.save('data/digit_test_x.npy', dataset.reshape((len(dataset), train_time_points * train_space_points, 3)))

# # np.save('data/digit_train_pos.npy', S_dataset.reshape((len(S_dataset), train_time_points * train_space_points, 1)))
# np.save('data/digit_test_pos.npy', S_dataset.reshape((len(S_dataset), train_time_points * train_space_points, 1)))

# # np.save('data/digit_train_time.npy', T_dataset.reshape((len(T_dataset), train_time_points * train_space_points, 1)))
# np.save('data/digit_test_time.npy', T_dataset.reshape((len(T_dataset), train_time_points * train_space_points, 1)))

np.save("data/digit_normalization.npy", normalization)

#############################################################
# Forward, Interpolation, Imputation Generation Test Set
#############################################################

# Num_of_variables = 3
# ds = .2
# dt = .008 # dt<ds**2 to keep the equations stable
# T_max = 10
# test_set = 500
# sampling_interval = dt
# S = np.arange( 0, 8, ds) # Space range
# T = np.arange( 0, T_max + dt, dt) # Time range
# num_nodes = 100
# S_size = len(S)
# T_size = len(T) - 1
# dataset_size = int(S_size * T_size / num_nodes)

# test_time_points = 10
# test_space_points = 20

# dataset = np.nan * np.ones([test_set, test_time_points, test_space_points, Num_of_variables])
# time_indices = np.arange(0, T_size, int(T_size / test_time_points), dtype=int)
# space_indices = np.arange(0, S_size, int(S_size / test_space_points), dtype=int)

# S_dataset = np.nan * np.ones([test_set, test_time_points, test_space_points, 1])
# T_dataset = np.nan * np.ones([test_set, test_time_points, test_space_points, 1])

# for test_i in trange(test_set):
#     X = np.random.uniform(-0.01, 0.01, [ len(S), Num_of_variables] )
#     X_plot = np.nan * np.ones( [ T_max, len(S), Num_of_variables] )
#     X_store = np.nan * np.ones( [ int(1/sampling_interval)*T_max, len(S), Num_of_variables] )

    
#     for i in range(1,len(T)):
#         # print(i)
#         X[1:-1] = X[1:-1] + dt * Velocity(X, ds)
#         # Neumann conditions: The derivatives at the edges are null
#         X[0] = X[1]
#         X[-1] = X[-2]

#         if (i%int(1/dt)==0):
#             # print(T[i])
#             X_plot[int(i/int(1/dt)-1)] = deepcopy(X)

#         if (i%int(1/dt*sampling_interval)==0):
#             X_store[int(i/int(1/dt*sampling_interval)-1)] = deepcopy(X)
    
#     dataset[test_i] = X_store[time_indices][:, space_indices]
    
#     S_dataset[test_i] = np.repeat(S[space_indices][np.newaxis, :], test_time_points, axis=0)[..., np.newaxis]
#     T_dataset[test_i] = np.dstack([T[time_indices]] * test_space_points)[..., np.newaxis]

# np.save('data/digit_test_conditional_x.npy', dataset)
# np.save('data/digit_test_conditional_pos.npy', S_dataset)
# np.save('data/digit_test_conditional_time.npy', T_dataset)

#############################################################
# Knockout Generation Test Set
#############################################################

# Num_of_variables = 3
# ds = .2
# dt = .008 # dt<ds**2 to keep the equations stable
# T_max = 10
# test_set = 500
# sampling_interval = dt
# S = np.arange( 0, 8, ds) # Space range
# T = np.arange( 0, T_max + dt, dt) # Time range
# num_nodes = 100
# S_size = len(S)
# T_size = len(T) - 1
# dataset_size = int(S_size * T_size / num_nodes)

# test_time_points = 10
# test_space_points = 20

# dataset = np.nan * np.ones([test_set, test_time_points, test_space_points, Num_of_variables])
# time_indices = np.arange(0, T_size, int(T_size / test_time_points), dtype=int)
# space_indices = np.arange(0, S_size, int(S_size / test_space_points), dtype=int)
# knockout_values = np.nan * np.ones([test_set])

# S_dataset = np.nan * np.ones([test_set, test_time_points, test_space_points, 1])
# T_dataset = np.nan * np.ones([test_set, test_time_points, test_space_points, 1])

# for test_i in trange(test_set):
#     knockout_i = np.random.choice(3, 1)
#     knockout_values[test_i] = knockout_i
#     X = np.random.uniform(-0.01, 0.01, [ len(S), Num_of_variables] )
#     X_plot = np.nan * np.ones( [ T_max, len(S), Num_of_variables] )
#     X_store = np.nan * np.ones( [ int(1/sampling_interval)*T_max, len(S), Num_of_variables] )

#     for i in range(1,len(T)):
#         # print(i)
#         X[:, knockout_i] = 0
#         X[1:-1] = X[1:-1] + dt * Velocity(X, ds)
#         # Neumann conditions: The derivatives at the edges are null
#         X[0] = X[1]
#         X[-1] = X[-2]

#         X[:, knockout_i] = 0

#         if (i%int(1/dt)==0):
#             # print(T[i])
#             X_plot[int(i/int(1/dt)-1)] = deepcopy(X)

#         if (i%int(1/dt*sampling_interval)==0):
#             X_store[int(i/int(1/dt*sampling_interval)-1)] = deepcopy(X)
    
#     dataset[test_i] = X_store[time_indices][:, space_indices]
#     S_dataset[test_i] = np.repeat(S[space_indices][np.newaxis, :], test_time_points, axis=0)[..., np.newaxis]
#     T_dataset[test_i] = np.dstack([T[time_indices]] * test_space_points)[..., np.newaxis]

# np.save('data/digit_test_knockout_x.npy', dataset)
# np.save('data/digit_test_knockout_pos.npy', S_dataset)
# np.save('data/digit_test_knockout_time.npy', T_dataset)
# np.save('data/digit_test_knockout_values.npy', knockout_values)