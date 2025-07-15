import numpy as np
from matplotlib import pyplot as plt
from model import Model, PiLearnModelUdelay
from setup.models import get_base_pars, get_feedback_model_pars


if __name__ == '__main__':
    P, G, mu, g = get_base_pars()
    # P, G, mu, g = get_feedback_model_pars()
    P, G, mu, g = get_feedback_model_pars(g12=5, mu3=20)
    dt = 1e-1

    model = Model(P, G, mu, dt=dt)

    T = 1000
    t, a, u = model.simulate(g, T=T)

    fig, axs = plt.subplots(3, 2, sharex=False)
    gs = axs[0,0].get_gridspec()
    for ax in axs[:,0]:
        ax.remove()
    ax = fig.add_subplot(gs[:,0])
    ax1, ax2, ax3 = axs[:,1]

    burnin = 500
    a, u, t = a[t>burnin], u[t>burnin], t[t>burnin]
    # a = np.clip(a, a_min=0, a_max=None)
    # u = np.clip(u, a_min=mu, a_max=None)
    x = np.concatenate([a,u], axis=-1)
    m = x.mean(axis=0)
    s = x.std(axis=0)
    x = x[:,s!=0]
    m = m[s!=0]
    s = s[s!=0]
    print(x.shape)
    cov = np.cov(x/s, rowvar=False)
    ev, evec = np.linalg.eig(cov)
    idx = np.argsort(ev)[::-1]
    ev = ev[idx]
    evec = evec[:, idx]
    proj = evec[:,:2]
    x_proj = ((x - m) / s) @ proj
    x_rec = (x_proj @ proj.T) * s + m
    print(proj.shape)
    ax.scatter(x_proj[:,0], x_proj[:,1], c='r', s=2**2)
    print(proj)
    ax1.plot(t/24, np.clip(x/s, a_min=0, a_max=None))
    ax2.plot(t/24, np.clip(x_rec/s, a_min=0, a_max=None))
    ax1.set_xlim(T/24-100/24, T/24)
    ax2.set_xlim(*ax1.get_xlim())
    ax3.bar([1,2], ev[:2], width=1, align='center', color='b')
    ax3.bar(np.arange(3, len(ev)+1), ev[2:], width=1, align='center',
            edgecolor='b', color='w')

    P_est = P + np.random.randn(*P.shape)
    learnmodel = PiLearnModelUdelay(P, G, mu, P_est=P_est, dt=dt, delay=50,
                                    eta=5e-4)
    t, a, u, _ = learnmodel.simulate(g, T=T/2)
    x = np.concatenate([a,u], axis=-1)
    x_proj = ((x - m) / s) @ proj
    ax.scatter(x_proj[:,0], x_proj[:,1], c=t, s=2**2, alpha=.4, zorder=10)

    plt.savefig('figures/manifold.png')
    plt.show()
