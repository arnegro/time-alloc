import numpy as np

def get_base_pars():
    N = 3
    P = np.zeros((N, N))
    P[0,0] = 4
    P[1,1] = 4
    P[2,:] = [-1, -1, 3]
    G = np.zeros((N, N))
    G[0,:] = [2, 3, 3]
    G[1,:] = [3, 4, 3]
    G[2,:] = [3, 3, 2]
    G *= 4
    mu = np.array([0, 20, 0])
    g = lambda t : np.array([4, 0, 0])
    return P, G, mu, g
