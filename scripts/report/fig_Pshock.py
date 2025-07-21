import pickle
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from model import Model, PiLearnModelUdelayProb, PiLearnModelUdelay
from setup.models import get_base_pars
from plotting import halfwidth, half_l, save_plot

force = False

U_inv = np.array([[29407.43824886,  7096.98509843, 12002.11559096],
                  [ 7096.98509843,  1825.97045419,  2931.03892697],
                  [12002.11559096,  2931.03892697,  5077.95203589]])
U_inv /= np.linalg.trace(U_inv)

dx = 2

def running_stats(y, window):
    means, q_low, q_high = [], [], []
    std = []
    for i in range(len(y)):
        if i < window:
            yw = y[:i+1]
        else:
            yw = y[i-window+1:i+1]
        means.append(np.mean(yw))
        # q_low.append(np.quantile(yw, 0.1))
        # q_high.append(np.quantile(yw, 0.9))
        std.append(np.std(yw))
    return np.array(means), np.array(std)

def run(model, g, T, a0, u0):
    t, a, _, _ = model.simulate(g=g, T=T, a0=a0.copy(),
                                u0=u0.copy(), uest0=u0.copy())
    w = np.cumsum(np.dot(np.clip(a, a_min=0, a_max=None), model.P[-1]))
    m, std = running_stats(w, k)
    return w, m, std, model.err

def run_all(P, G, mu, g, T1, T, max_certs, dt=1e-1, delay=50):
    basemodel = Model(P=P, G=G, mu=mu, dt=dt)
    t1, a1, u1 = basemodel.simulate(g=g, T=T1)
    a0, u0 = a1[-1], u1[-1]
    basemodel.P[-1,0] *= 2
    t2, a2, _ = basemodel.simulate(g=g, T=T, a0=a0, u0=u0)
    w1 = np.cumsum(np.dot(np.clip(a1, a_min=0, a_max=None), P[-1]))
    w2 = np.cumsum(np.dot(np.clip(a2, a_min=0, a_max=None), basemodel.P[-1]))
    res = {}
    res['base'] = {}
    res['base']['pre'] = (t1, w1)
    res['base']['post'] = (t2, w2)

    gdmodel = PiLearnModelUdelay(P=P, G=G, mu=mu, P_est=P, dt=dt,
                                 delay=delay, eta=1e-4)
    gdmodel.P[-1,0] *= 2
    res['gd'] = run(gdmodel, g, T, a0, u0)

    res['bayes'] = {}
    for max_cert in max_certs:
        bayesmodel = PiLearnModelUdelayProb(P=P, G=G, mu=mu, P_est=P, dt=dt,
                                            delay=delay, U_inv=U_inv*max_cert)
        bayesmodel.P[-1,0] *= 2
        res['bayes'][max_cert] = run(bayesmodel, g, T, a0, u0)

    return res

def plot(res, axs):
    max_certs = list(res['bayes'].keys())
    ms = []
    t1, w1 = res['base']['pre']
    m1 = np.mean(w1[t1>t1.max()-300])
    t2, w2 = res['base']['post']
    mask = t2>t2.max()-1000
    wgd, _, _, errgd = res['gd']
    axs[0].hlines(wgd[mask].mean()+m1, 0, 1, ls='-', color='r', lw=.5,
                  transform=axs[0].get_yaxis_transform(), label='GD')
    axs[1].hlines(errgd[-1], 0, 1, ls='-', color='r', lw=.5,
                  transform=axs[1].get_yaxis_transform())
    axs[0].hlines(w2[mask].mean()+m1, 0, 1, ls='--', color='k', lw=.5,
                  transform=axs[0].get_yaxis_transform(),
                  label=r'$\Pi$ known')
    for lab, ax in zip(['i', 'ii'], axs):
        ax.grid(lw=.1)
        ax.text(.01, .97, f'(c{lab})', transform=ax.transAxes,
                va='top', ha='left', size='xx-small')
    for mc, (w, m, std, err) in res['bayes'].items():
            w = w + m1
            m = np.mean(w[mask])
            q1, q2 = np.quantile(w[mask], [.1, .9])
            axs[0].errorbar(mc, m, yerr=([m-q1], [q2-m]), zorder=100,
                            c='k', capsize=2, fmt='-o', markersize=1.75**2,
                            lw=.5, capthick=.5)
            ms.append((mc, err[-1], m))
    ms = sorted(ms, key=lambda mcm: mcm[0])
    mcs, errs, ms = zip(*ms)
    axs[0].plot(mcs, ms, zorder=10, lw=.75, c='k')
    axs[-1].plot(mcs, errs, zorder=10, lw=.75, c='k')
    axs[-1].set(xlabel=r'$\mathrm{tr}(U^{-1}_0)$', xscale='log', yscale='log',
                xlim=(min(max_certs)/dx, max(max_certs)*dx))
    axs[0].legend(ncols=2, frameon=False, fontsize='xx-small',
                  loc='lower right', bbox_to_anchor=[0,1.05,1,.5],
                  borderaxespad=0)

def plot_trajectories(res, axs, cax_parent, ax):
    t1, w1 = res['base']['pre']
    t2, w2 = res['base']['post']
    t1 = t1/24/30
    t2 = t2/24/30
    t1 -= t1.max() + np.diff(t1).mean()
    # t2 += t1[-1] + np.diff(t1).mean()
    m1 = np.mean(w1[t1>-300/24/30])
    m2 = np.mean(w2[t2>300/24/30])
    m2 += m1
    axs[0].plot(t1, [m1]*len(t1), c='k', ls='--', lw=.5)
    axs[0].plot([t1[-1], t2[0]], [m1, m2], c='k', ls='--', lw=.5)
    axs[0].plot(t2, [m2]*len(t2), c='k', ls='--', lw=.5)
    ax.plot(t2, [m2]*len(t2), c='k', ls='--', lw=.5)
    for lab, _ax in zip(['i', 'ii'], axs):
        _ax.grid(lw=.1)
        _ax.text(.01, 1-.97, f'(b{lab})', transform=_ax.transAxes,
                 va='bottom', ha='left', size='xx-small')
    ax.text(.01, .9, f'(a)', transform=ax.transAxes,
            va='top', ha='left', size='x-small')
    # axs[1].plot(t1, [s1]*len(t1), c='k')
    # axs[1].plot([t1[-1], t2[0]], [s1, s2], c='k')
    # axs[1].plot(t2, [s2]*len(t2), c='k')

    wgd, mgd, sgd, errgd = res['gd']
    axs[0].plot(t2, mgd+m1, c='r', lw=.5, zorder=1000)
    # _k = k//5
    # sgd[:_k] = np.interp(range(_k), [0,k], [s1, sgd[_k]])
    # axs[1].plot(t2, sgd, c='r')
    axs[1].plot(t2, errgd, c='r', lw=.5, zorder=1000)

    max_certs = list(res['bayes'].keys())
    max_certs.sort()
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=min(max_certs)/dx, vmax=max(max_certs)*dx)
    get_color = lambda s : cmap(norm(s))
    for max_cert, (wb, mb, sb, errb) in res['bayes'].items():
        axs[0].plot(t2, mb+m1, c=get_color(max_cert), lw=.5)
        # _k = k//5
        # sb[:_k] = np.interp(range(_k), [0,k], [s1, sb[_k]])
        # axs[1].plot(t2, sb, c=get_color(max_cert), lw=.5)
        axs[1].plot(t2, errb, c=get_color(max_cert), lw=.5)
        if max_cert in max_certs[::2]:
            ax.plot(t2, wb+m1, c=get_color(max_cert), lw=.5)

    axs[1].set(yscale='log', xlabel='time [months]', ylabel='MSE')
    axs[0].set(ylabel=r'$\omega$ [€]')
    cax = cax_parent.inset_axes([0, 1, 1, .05])
    im = cax.imshow([[]], aspect='auto', cmap=cmap, norm=norm)
    plt.colorbar(im, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.set(xticks=[])

    ax.set(xticks=[], yticks=[], xlim=(t2.min(), t2.min()+2))
    d = 7
    ax.plot([t2.min(), t2.min()+d/30], [0, 0], c='k', lw=2,
            transform=ax.get_xaxis_transform())
    ax.text(t2.min() + d/2/30, -.05, f'{d} days', size='xx-small',
            ha='center', va='top', transform=ax.get_xaxis_transform())
    y0, _ = ax.get_ylim()
    dw = 500
    ax.plot([0, 0], [y0, y0+dw], c='k', lw=2,
            transform=ax.get_yaxis_transform())
    ax.text(-0.01, y0+dw/2, f'{dw}€', size='xx-small', ha='right',
            va='center', transform=ax.get_yaxis_transform())
    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == '__main__':
    P, G, mu, g = get_base_pars(pi=-1)

    dt = 1e-1
    delay = 50
    T1 = 1000
    T = 8000
    k = int(30*24/dt)

    # max_certs = [1e1, 1e3, 1e4, 1e5, 1e6, 1e7]
    max_certs = np.geomspace(1e1, 1e12, 10)

    pickle_fl = Path('data') / 'P_shock.pickle'
    if not force and pickle_fl.exists():
        with pickle_fl.open('rb') as fl:
            res = pickle.load(fl)
    else:
        res = run_all(P, G, mu, g, T1, T, max_certs, dt=dt, delay=delay)
        with pickle_fl.open('wb') as fl:
            pickle.dump(res, fl)

    fig, axs = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(halfwidth, 2.5))
    gs = axs[0,0].get_gridspec()
    for ax in axs[0]:
        ax.remove()
    ax = fig.add_subplot(gs[0,:])
    ax.patch.set_alpha(0)

    plot_trajectories(res, axs[1:,0], axs[1,1], ax)     
    plot(res, axs[1:,1])     

    fig.subplots_adjust(left=half_l, top=1.0, bottom=0.145, right=0.989,
                        hspace=0.236, wspace=0.1)
    save_plot('fig_Pshock')
    plt.show()
