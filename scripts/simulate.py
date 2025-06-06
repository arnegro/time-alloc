import numpy as np
from matplotlib import pyplot as plt
from model import Model, PiLearnModelU, PiLearnModelDU, PiLearnModelUdelay

if __name__ == '__main__':
    n = 3
    P = np.zeros((n, n))
    P[0,0] = 4
    P[1,1] = 4
    P[2,:] = [-1, -1, 3]
    G = np.zeros((n, n))
    G[0,:] = [2, 3, 3]
    G[1,:] = [3, 4, 3]
    G[2,:] = [3, 3, 2]
    G *= 4
    mu = np.array([0, 20, 0])
    g = lambda t : np.array([4, 0, 0])

    P_est = P + np.random.randn(*P.shape)*1e-2
    model = PiLearnModelUdelay(P, G, mu, P_est=P_est, delay=20)
    model = PiLearnModelU(P, G, mu, P_est=P_est, dt=1e-1)
    print(model.P)
    print(np.round(model.P_est, decimals=3))
    t, a, u, u_est = model.simulate(g, T=1000)
    print(np.round(model.P_est, decimals=3))
    # model = Model(P, G, mu)
    # t, a, u = model.simulate(g, T=100)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    ax1.plot(t, np.clip(a, a_min=0, a_max=None))
    ax2.plot(t, np.clip(u, a_min=mu, a_max=None))
    # ax2.plot(t, u)
    ax3.plot(t, np.clip(u_est, a_min=mu, a_max=None),
             label=['HH', 'leisure', 'work'])
    ax4.plot(t, model.err)
    ax1.set(ylabel=r'$\vec{a}^{> 0}$')
    # ax2.set(ylabel=r'$\vec{u}^{> \vec{\mu}}$')
    ax2.set(ylabel=r'$\vec{u}$')
    y0, y1 = ax3.get_ylim()
    ax3.set(xlabel='time', ylabel=r'$\vec{u}_{est}^{> \vec{\mu}}$',
            ylim=(y0-.1*(y1-y0), y1))
    ax3.legend(ncols=3, frameon=False)
    plt.tight_layout()
    plt.show()
