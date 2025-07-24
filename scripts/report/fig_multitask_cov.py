import numpy as np
from matplotlib import pyplot as plt
from model import Model
from setup.models import get_base_pars
from plotting import save_plot, halfwidth, half_l, despine

def get_corrs(gammas, T, dt):
    corr = []
    acts = []
    for gamma in gammas:
        P, G, mu, g = get_base_pars(gamma=gamma)
        model = Model(P, G, mu, dt=dt)
        t, a, u = model.simulate(g, T=T)
        _a = np.clip(a, a_min=0, a_max=None)
        corr.append(np.corrcoef(_a[:,:2], rowvar=False)[1,0])
    return corr


if __name__ == '__main__':
    gamma = np.linspace(-1, 3, 10)
    n = len(gamma)
    ex_gammas = [0, 2.5, 3]
    T = 1000
    dt = 5e-2
    corr = get_corrs(gamma, T, dt=dt)
    fig, axs = plt.subplots(3, 2, sharex='col', figsize=(halfwidth, 2))
    gs = axs[0,0].get_gridspec()
    for ax in axs[:,0]:
        ax.remove()
    ax = fig.add_subplot(gs[:,0])
    ax.plot(gamma, corr, c='k')
    ax.plot([0, 1], [0, 0], transform=ax.get_yaxis_transform(), lw=.3,
            c='k', ls='--')
    ax.set(xlabel='$\gamma$', ylabel='corr. of HH and leisure activity')
    ax.text(0.02, 0.02, f'(a)', transform=ax.transAxes, size='small',
            ha='left', va='bottom')
    for _ax, gam, lab in zip(axs[:,1], ex_gammas, 'bcdef'):
        P, G, mu, g = get_base_pars(gamma=gam)
        model = Model(P, G, mu, dt=dt)
        t, a, u = model.simulate(g, T=200)
        a = np.clip(a, a_min=0, a_max=None)
        _ax.plot(t, a[:,0], lw=.75, label='HH')
        _ax.plot(t, a[:,1], lw=.75, label='leisure')
        _ax.plot(t, a[:,2], lw=.5, label='work')
        _ax.text(0, 1, f'({lab})', transform=_ax.transAxes, size='x-small')
        ax.plot([gam, gam], [0, 1], transform=ax.get_xaxis_transform(), lw=.3,
                c='k', ls=':')
        ax.text(gam, 0.9, lab, ha='center', size='xx-small',
                transform=ax.get_xaxis_transform(),
                bbox=dict(facecolor='white', edgecolor='none', pad=1))
        _ax.set(xticks=[], yticks=[])
        despine(_ax, 'all')
    axs[-1,1].set(xlim=(150, 200))
    axs[-1,1].plot([180, 200], [-.1]*2, c='k', lw=2, clip_on=False,
                   transform=axs[-1,1].get_xaxis_transform())
    axs[-1,1].text(190, -.18, '20h', va='top', ha='center',
                   size='xx-small',
                   transform=axs[-1,1].get_xaxis_transform())
    fig.tight_layout()
    axs[-1,1].legend(loc='lower right', ncols=2, bbox_to_anchor=[0,-1.5,1,1],
                     fontsize='xx-small', frameon=False)
    fig.subplots_adjust(left=half_l)
    save_plot('fig_multitask_cov')
    plt.show()
