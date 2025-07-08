import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib import cm
from model import FeedbackModelG, FeedbackModelPi, FeedbackModelMu

def clip(x, y=0):
    return max(x, y)

def beta_plot():
    n = 100
    fig, ax = plt.subplots()
    betas = np.linspace(-.5, 1, n)
    cmap = plt.get_cmap('plasma')
    ## set x of divergence to one: * pi at x
    ax.plot(betas, 1 / (1 - betas), label=r'$g$')
    ## set x of divergence to one: * pi^2/4g at x
    ax.plot(betas, 2 / (1 + np.sqrt(1 - betas)), label=r'$\Pi$')
    ax.set(yscale='log', xlabel=r'$\beta / \beta_{c}$',
            ylabel=r'$a^* / a^*_0$',
            title=r'rescaling of action fixed point under different $\beta$')
    plt.grid()
    ax.legend(title='feedback via')
    plt.savefig('figures/feedback_1d-beta-rescale-a.png')
    plt.show()

if __name__ == '__main__':

    beta_plot()
    quit()

    pi, mu, g_star = 1, 0, 0
    gam = 2
    beta, tau = 0.89, 3

    G = gam*np.ones((1,1))
    P = pi*np.ones((1,1))
    _mu = mu*np.ones(1)
    _g = g_star*np.ones(1)

    fu = lambda a,u,g : g - pi*clip(a)
    fa = lambda a,u,g : clip(u,mu) - gam*clip(a)
    fg = lambda a,u,g : (g_star - g + beta*clip(a)) / tau
    
    g0s = np.linspace(0, 1, 5)
    cmap2 = lambda g0 : cm.plasma((g0 - g0s.min()) / (g0s.max() - g0s.min()))

    """
    fig, ax = plt.subplots(1)
    a = np.arange(0, 5, 1e-1)
    u = np.arange(0, 5, 1e-1)
    for g in g0s:
        ax.plot(a, g/pi*np.ones_like(a), c=cmap2(g))
        ax.plot(a, u/gam, c=cmap2(g))
    """


    fig, axs = plt.subplots(3, 3, figsize=(8,8))
    for g0 in g0s:
        # _g0 = g0*np.ones(1)
        _g0 = .01 #g0/10
        _b = g0/10
        model = FeedbackModelG(P=P, mu=_mu, G=G, g=_g+_g0,
                               beta=beta+_b, tau=tau,
                               dt=1e-1, sigma_u=.0)
        # model.g = _g0
        t, a, u, g = model.simulate(T=500)
        res = [a, u, g]
        labels = ['a', 'u', 'g']
        for i, x in enumerate(res):
            axs[i,i].plot(t, x, c=cmap2(g0))
            axs[i,i].set_ylabel(labels[i])
            for j in range(i+1,len(res)):
                y = res[j]
                ax = axs[i,j]
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                colors = cm.viridis((t[:-1] - t.min()) / (t.max() - t.min()))
                lc = LineCollection(segments, colors=colors, linewidth=2)
                ax.add_collection(lc)
                axs[i,j].set(xlabel=labels[i], ylabel=labels[j],
                             xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
    ax = axs[-1,-1]
    ax.plot([0, 1], [g_star / (1 - beta/pi)]*2,
            transform=ax.get_yaxis_transform(), c='k', ls='--')
    print(g_star/pi)


    fig.tight_layout()
    plt.show()
