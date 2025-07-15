import numpy as np

def get_base_pars(gamma=3):
    N = 3
    P = np.zeros((N, N))
    P[0,0] = 4
    P[1,1] = 4
    P[2,:] = [-1, -1, 3]
    G = np.zeros((N, N))
    G[0,:] = [2, gamma, 3]
    G[1,:] = [gamma, 4, 3]
    G[2,:] = [3,     3, 2]
    G *= 4
    mu = np.array([0, 20, 0])
    g = lambda t : np.array([4, 0, 0])
    return P, G, mu, g

def get_feedback_model_pars(g12=10, g23=10, g31=10, mu3=0):
    mu = np.array([0,0,mu3])
    P = np.array([[4,0,-2], [-2, 4, 0], [0, -2, 4]])
    G = np.array([[8,g12,10], [10, 8, g23], [g31, 10, 8]])
    g = lambda t : np.array([2, 0, 0])
    return P, G, mu, g

