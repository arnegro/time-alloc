import pickle
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import types
from model import Model, PiLearnModelUdelayProb, PiLearnModelUdelay
from setup.models import get_base_pars
from plotting import halfwidth, half_l, save_plot

force = False

U_inv = np.array([[29407.43824886,  7096.98509843, 12002.11559096],
                  [ 7096.98509843,  1825.97045419,  2931.03892697],
                  [12002.11559096,  2931.03892697,  5077.95203589]])
U_inv /= np.linalg.trace(U_inv)

dx = 2

def get_record(P_adj):
    def _rec(self, t):
        update = t // self.delay != (t+self.dt) // self.delay
        if update:
            self.err.append((t, (self.u_est[0] - self.u[0])**2))
    return _rec

def running_stats(y, window):
    means, q_low, q_high = [], [], []
    std = []
    for i in range(len(y)):
        if i < window:
            yw = y[:i+1]
        else:
            yw = y[i-window+1:i+1]
        means.append(np.max(yw))
        # means.append(np.mean(yw[yw>0]))
        # q_low.append(np.quantile(yw, 0.1))
        # q_high.append(np.quantile(yw, 0.9))
        std.append(np.std(yw))
    return np.array(means), np.array(std)

def run(model, g, T, a0, u0, ma):
    P_adj = - np.outer(model.g_bias, ma) / np.dot(ma,ma)
    model._record = types.MethodType(get_record(P_adj), model)
    t, a, _, _ = model.simulate(g=g, T=T, a0=a0.copy(),
                                u0=u0.copy(), uest0=u0.copy())
    # w = np.cumsum(np.dot(np.clip(a, a_min=0, a_max=None), model.P[-1]))
    w = np.clip(a[:,0], a_min=0, a_max=None)
    m, std = running_stats(w, k)
    t_err, err = zip(*model.err)
    t_err, err = np.array(t_err), np.array(err)
    err, _ = running_stats(err, 4)
    return w, m, std, t_err, err

def run_all(P, G, mu, g, T1, T, max_certs, g_bias, g2, dt=1e-1, delay=50):
    basemodel = Model(P=P, G=G, mu=mu, dt=dt)
    t1, a1, u1 = basemodel.simulate(g=g, T=T1)
    a0, u0 = a1[-1], u1[-1]
    t2, a2, u2 = basemodel.simulate(g=g2, T=T, a0=a0, u0=u0)
    w1 = np.clip(a1[:,0], a_min=0, a_max=None)
    w2 = np.clip(a2[:,0], a_min=0, a_max=None)
    # w1 = np.cumsum(np.dot(np.clip(a1, a_min=0, a_max=None), P[-1]))
    # w2 = np.cumsum(np.dot(np.clip(a2, a_min=0, a_max=None), basemodel.P[-1]))
    res = {}
    res['base'] = {}
    res['base']['pre'] = (t1, w1)
    res['base']['post'] = (t2, w2)

    gdmodel = PiLearnModelUdelay(P=P, G=G, mu=mu, P_est=P, dt=dt,
                                 delay=delay, eta=1e-4)
    gdmodel.g_bias = g_bias
    ma = np.mean(np.clip(a2[t2>300], a_min=0, a_max=None), axis=0)
    res['gd'] = run(gdmodel, g2, T, a0, u0, ma)

    res['bayes'] = {}
    for max_cert in max_certs:
        bayesmodel = PiLearnModelUdelayProb(P=P, G=G, mu=mu, P_est=P, dt=dt,
                                            delay=delay, U_inv=U_inv*max_cert)
        bayesmodel.g_bias = g_bias
        res['bayes'][max_cert] = run(bayesmodel, g2, T, a0, u0, ma)

    return res

def plot(res, axs):
    max_certs = list(res['bayes'].keys())
    ms = []
    t1, w1 = res['base']['pre']
    m1 = np.mean(w1[t1>t1.max()-300])
    t2, w2 = res['base']['post']
    mask = t2>t2.max()-1000
    axs[0].hlines(np.max(w2[mask]), 0, 1, ls='--', color='k', lw=.5,
                  transform=axs[0].get_yaxis_transform(),
                  label=r'$\Pi$ known', zorder=-200)
    axs[0].hlines(np.median(w2[mask][w2[mask]>0]), 0, 1, ls='--', color='k',
                  lw=.5, transform=axs[0].get_yaxis_transform(), zorder=-200)
    wgd, _, _, _, errgd = res['gd']
    axs[0].hlines(wgd[mask].max(), 0, 1, ls='-', color='r', lw=.5,
                  transform=axs[0].get_yaxis_transform(), label='GD (max)',
                  zorder=-200)
    axs[0].hlines(np.median(wgd[mask]), 0, 1, ls='-', color='r', lw=.5,
                  transform=axs[0].get_yaxis_transform(), zorder=-200)
    axs[1].hlines(errgd[-1], 0, 1, ls='-', color='r', lw=.5,
                  transform=axs[1].get_yaxis_transform())
    for lab, ax in zip(['i', 'ii'], axs):
        ax.grid(lw=.1)
        ax.text(.01, .97, f'(c{lab})', transform=ax.transAxes,
                va='top', ha='left', size='xx-small')
    violinax = axs[0].twiny()
    for mc, (w, m, std, _, err) in res['bayes'].items():
            w = w[mask]
            m = np.median(w[w>0])
            vp = violinax.violinplot([w[w>0]], positions=[np.log10(mc)],
                                     widths=.5)
            # Set linewidth for violin bodies
            for body in vp['bodies']:
                body.set_edgecolor('black')
                body.set_facecolor('lightgray')
                body.set_linewidth(.2)      # Set outline thickness
                body.set_alpha(1)        # Optional: adjust transparency
                body.set_zorder(90)

            # Set linewidth for other parts (center bar, mins, maxes, medians)
            for partname in ['cmins', 'cmaxes', 'cbars', 'cmedians', 'cmeans']:
                if partname in vp:
                    vp[partname].set_linewidth(.2)
                    vp[partname].set_color('black')
                    vp[partname].set_zorder(100)

            """
            q1, q2 = np.quantile(w[mask], [0, 1])
            axs[0].errorbar(mc, m, 
                            yerr=([m-q1], [q2-m]),
                            # yerr=np.std(w[mask]),
                            zorder=100,
                            c='k', capsize=2, fmt='-o', markersize=1.75**2,
                            lw=.5, capthick=.5)
            """
            ms.append((mc, err[-1], m))
    ms = sorted(ms, key=lambda mcm: mcm[0])
    mcs, errs, ms = zip(*ms)
    # axs[0].plot(mcs, ms, zorder=10, lw=.75, c='k')
    violinax.plot(np.log10(mcs), ms, zorder=500, lw=.5, c='k', ls='-')
    axs[0].set(yscale='log', ylim=(.3, axs[0].get_ylim()[1]))
    axs[-1].plot(mcs, errs, zorder=10, lw=.75, c='k')
    axs[-1].set(xlabel=r'$\mathrm{tr}(U^{-1}_0)$', xscale='log', yscale='log',
                xlim=(min(max_certs)/dx, max(max_certs)*dx))
    axs[0].legend(ncols=3, frameon=False, fontsize=6,
                  loc='lower right', bbox_to_anchor=[0,1.05,1,.5],
                  borderaxespad=0)
    violinax.set(xticks=[], xlim=np.log10(axs[0].get_xlim()))

def plot_trajectories(res, axs, cax_parent, ax):
    t1, w1 = res['base']['pre']
    t2, w2 = res['base']['post']
    t1 = t1/24/30
    t2 = t2/24/30
    t1 -= t1.max() + np.diff(t1).mean()
    # t2 += t1[-1] + np.diff(t1).mean()
    m1 = np.max(w1[t1>-300/24/30])
    m2 = np.max(w2[t2>300/24/30])
    # m2 += m1
    axs[0].plot(t1, [m1]*len(t1), c='k', ls='--', lw=.5)
    axs[0].plot([t1[-1], t2[0]], [m1, m2], c='k', ls='--', lw=.5)
    axs[0].plot(t2, [m2]*len(t2), c='k', ls='--', lw=.5)
    ax.plot(t2, [m2]*len(t2), c='k', ls='--', lw=.5)
    # axs[1].plot(t1, [s1]*len(t1), c='k')
    # axs[1].plot([t1[-1], t2[0]], [s1, s2], c='k')
    # axs[1].plot(t2, [s2]*len(t2), c='k')

    wgd, mgd, sgd, terrgd, errgd = res['gd']
    l = np.argmin(mgd < m1)
    mgd[:l] = m1
    errgd, _ = running_stats(errgd, 20)
    axs[0].plot(t2, mgd, c='r', lw=.5, zorder=1.5)
    ax.plot(t2, wgd, c='r', ls='-', lw=.5)
    # _k = k//5
    # sgd[:_k] = np.interp(range(_k), [0,k], [s1, sgd[_k]])
    # axs[1].plot(t2, sgd, c='r')
    axs[1].plot(terrgd/24/30, errgd, c='r', lw=.5, zorder=1.5)

    max_certs = list(res['bayes'].keys())
    max_certs.sort()
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=min(max_certs)/dx, vmax=max(max_certs)*dx)
    get_color = lambda s : cmap(norm(s))
    for max_cert, (wb, mb, sb, terrb, errb) in res['bayes'].items():
        # _k = k//5
        # sb[:_k] = np.interp(range(_k), [0,k], [s1, sb[_k]])
        # axs[1].plot(t2, sb, c=get_color(max_cert), lw=.5)
        if max_cert in max_certs[::2]:
            errb, _ = running_stats(errb, 10)
            l = np.argmin(mb < m1)
            mb[:l] = m1
            axs[0].plot(t2, mb, c=get_color(max_cert), lw=.5)
            axs[1].plot(terrb/24/30, errb, c=get_color(max_cert), lw=.5)
        if max_cert in max_certs[::4]:
            ax.plot(t2, wb, c=get_color(max_cert), lw=.5)

    axs[1].set(yscale='log', xlabel='time [months]', ylabel='MSE$(u_{HH})$')
    axs[0].set(ylabel=r'HH intensity')
    axs[0].text(.99, .99, r'$\max$', size='xx-small', ha='right', va='top',
                transform=axs[0].transAxes)
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
    dw = 5
    ax.plot([0, 0], [y0, y0+dw], c='k', lw=2,
            transform=ax.get_yaxis_transform())
    ax.text(-0.01, y0+dw/2, f'{dw} HH', size='xx-small', ha='right',
            va='center', transform=ax.get_yaxis_transform())
    for spine in ax.spines.values():
        spine.set_visible(False)
    for lab, _ax in zip(['i', 'ii'], axs):
        _ax.grid(lw=.1)
        _ax.text(.015, 1-.97, f'(b{lab})', transform=_ax.transAxes,
                 va='bottom', ha='left', size='xx-small',
                 bbox=dict(fc='w', lw=0, pad=1.2), zorder=2)
    ax.text(.01, .9, f'(a)', transform=ax.transAxes,
            va='top', ha='left', size='x-small')


if __name__ == '__main__':
    P, G, mu, g = get_base_pars()
    g_bias = np.array([2, 0, 0])
    g2 = lambda t : g(t) + g_bias

    dt = 1e-1
    delay = 50
    T1 = 500
    T = 10000
    k = int(7*24/dt)

    # max_certs = [1e1, 1e3, 1e4, 1e5, 1e6, 1e7]
    max_certs = np.geomspace(1e1, 1e12, 20)

    pickle_fl = Path('data') / 'g_shock.pickle'
    if not force and pickle_fl.exists():
        with pickle_fl.open('rb') as fl:
            res = pickle.load(fl)
    else:
        res = run_all(P, G, mu, g, T1, T, max_certs, g_bias, g2,
                      dt=dt, delay=delay)
        with pickle_fl.open('wb') as fl:
            pickle.dump(res, fl)

    fig, axs = plt.subplots(3, 2, sharex='col', sharey='row',
                            figsize=(halfwidth, 2.5))
    gs = axs[0,0].get_gridspec()
    for ax in axs[0]:
        ax.remove()
    ax = fig.add_subplot(gs[0,:])
    ax.patch.set_alpha(0)

    plot_trajectories(res, axs[1:,0], axs[1,1], ax)     
    plot(res, axs[1:,1])     

    fig.subplots_adjust(left=half_l, top=1.0, bottom=0.145, right=0.989,
                        hspace=0.236, wspace=0.1)
    save_plot('fig_gshock')
    plt.show()
