import numpy as np
from matplotlib import pyplot as plt
from model import Model, PiLearnModelU, PiLearnModelDU, PiLearnModelUdelay, PiLearnModelUdelayProb
from setup.models import get_base_pars

if __name__ == '__main__':

    P, G, mu, g = get_base_pars()

    P_est = P + np.random.randn(*P.shape)*1
    # model = Model(P, G, mu)
    # model = PiLearnModelUdelay(P, G, mu, P_est=P_est, delay=50, eta=1e-4, sigma_u=None)
    model = PiLearnModelUdelayProb(P, G, mu, P_est=P_est, delay=25, sigma_u=.5)
    # model = PiLearnModelU(P, G, mu, P_est=P_est, dt=1e-1)
    print(model.P)
    print(np.round(model.P_est, decimals=3))
    t, a, u, u_est = model.simulate(g, T=300)
    print(np.round(model.M, decimals=3))
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
