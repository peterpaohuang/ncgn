import math

#########################################################################
# KNN based schedules for DMP
#########################################################################

def log_schedule(t, k, N, curvature=10):
    start_clusters = math.sqrt(k * N)
    end_clusters = N

    n_clusters = (end_clusters - start_clusters) / math.log(curvature + 1) * math.log(curvature * t + 1) + start_clusters
    n_edges = (k * N) / n_clusters

    return math.ceil(n_clusters), math.ceil(n_edges)

def exp_schedule(t, k, N, curvature=10):
    """
    Calculate connectivity and coarse graining resolution based on exponetial curve 
    """

    start_clusters = math.sqrt(k * N)
    end_clusters = N

    n_clusters = (end_clusters - start_clusters) * ((math.exp(curvature * t) - 1)/(math.exp(curvature) - 1)) + start_clusters
    n_edges = (k * N) / n_clusters

    return math.ceil(n_clusters), math.ceil(n_edges)

def linear_schedule(t, k, N):
    """
    Calculate connectivity and coarse graining resolution based on linear interpolation
    """
    start_clusters = math.sqrt(k * N)
    end_clusters = N

    n_clusters = math.ceil((1 - t) * start_clusters + t * end_clusters)
    n_edges = (k * N) / n_clusters

    return n_clusters, math.ceil(n_edges)

def relu_schedule(t, k, N, split=0.6):
    """
    Calculate connectivity and coarse graining resolution based on relu
    """
    start_clusters = math.sqrt(k * N)
    end_clusters = N

    if t <= split:
        n_clusters = start_clusters
    else:
        slope = (end_clusters - start_clusters) / (1 - split)
        n_clusters = start_clusters + slope * (t - split)
    
    n_clusters = math.ceil(n_clusters)
    n_edges = (k * N) / n_clusters

    return n_clusters, math.ceil(n_edges)