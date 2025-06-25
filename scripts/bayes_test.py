import numpy as np
from matplotlib import pyplot as plt
from setup.models import get_base_pars

if __name__ == '__main__':
    # Dimensions
    p = 3  # number of features
    q = 3  # output dimension

    # Data (1 data point)
    M = np.zeros((q, p))
    U = np.eye(p)*1e2
    V = np.eye(q)
    Sigma = V  # likelihood noise = column covariance
    u = np.random.randn(q, 1)*0
    a = np.random.randn(p, 1)*0
    dt = 1e-2
    g = np.zeros_like(u)
    G = np.ones((q,q))
    P, G, mu, g = get_base_pars()
    g, mu = g(0)[:,None], mu[:,None]
    print(P)
    xs, ys = [], []
    for _ in range(30):
        # print(M)
        # input()
        u_prev = u.copy()
        _A = 0
        _G = 0
        f = lambda x : np.clip(x, a_min=0, a_max=None)
        u_s, a_s = [u[:,0]], [a[:,0]]
        for t in np.arange(0, 25, dt):
            u += dt * (g - P @ f(a))
            a += dt * (np.clip(u, a_min=mu, a_max=None) - G @ f(a))
            # print(u[:,0], a[:,0])
            _G += dt * g
            _A += dt * f(a)
            u_s.append(u[:,0].copy())
            a_s.append(a[:,0].copy())
        for t in np.arange(0, 25, dt):
            u += np.sqrt(dt) * np.random.randn() * .5
        u_s, a_s = np.array(u_s), np.array(a_s)
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # ax1.plot(np.clip(u_s, a_min=mu[:,0], a_max=None))
        # ax2.plot(f(a_s))
        # plt.show()
        
        x = _A
        # y = np.random.randn(q, 1)
        y = u - u_prev - _G
        # print(x)
        # print(y)
        # quit()
        xs.append(x)
        ys.append(y)
        # print('A', x)
        # print('u', y)

        # print(x)
        print(y)
        print(u)
        print(u_prev)
        print(_G)


        # Transform: y = A * (-x) + noise => define x_tilde
        x_tilde = -x  # shape (p, 1)

        # Prior: A ~ MN(M, U, V)

        # Compute posterior
        U_inv = np.linalg.inv(U)
        U_post = np.linalg.inv(U_inv + x_tilde @ x_tilde.T)  # shape (p, p)

        M_post = M + (y - M @ x_tilde).dot(x_tilde.T).dot(U_post)  # shape (q, p)
        # print(np.linalg.inv(U_post))
        # print(M)
        # print(M_post)
        input()
        # print(M_post)
        # quit()
        M = M_post
        U = U_post

        # Output
        # print("Posterior mean of A:\n", M_post)
        # input()
        # print("Posterior row covariance U_post:\n", U_post)
        # print("Posterior column covariance V_post:\n", Sigma)
    print("Posterior mean of A:\n", M)
    plt.scatter(xs, ys)
    plt.show()


