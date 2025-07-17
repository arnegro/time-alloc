from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
import pickle
from model import Model
from setup.models import get_base_pars, get_feedback_model_pars
from setup.model_comparison \
            import get_error_comparison, get_error_comparison_du, get_error_comparison_multitask
from plotting import halfwidth, save_plot, half_l

path = Path('figures')
force = False

def get_onsets(a):
    return np.where(np.diff((a > 0).astype(int), axis=0) > .5)[0]

def smooth(x, dt, n=7):
    n = int(n/dt)
    ker = np.ones(n) / n
    n *= 5
    _x = np.concatenate([[x[0]]*n, x, [x[-1]]*n])
    return np.convolve(_x, ker, mode='same')[n:-n]

def get_durations(ta, ons, offs):
    dt = np.diff(ta).mean()
    if offs[0] < ons[0]:
        offs = offs[1:]
    ons = ons[:len(offs)]
    durs = ta[offs] - ta[ons]
    return durs

def run(arg):
    name, (cls, kwargs) = arg
    t_sample = int(kwargs['delay']/dt) if 'delay' in kwargs else int(1/dt)
    err, Pe_err = [], []
    durs = []
    default_kwargs = dict(P=P, G=G, mu=mu, dt=dt)
    kwargs = default_kwargs | kwargs
    _P = kwargs['P']
    for j in range(n):
        print(f'[{j+1}/{n}] {name}', end='    \r')
        P_est = _P + np.random.randn(*_P.shape)*sd
        model = cls(P_est=P_est, **kwargs)
        t, a, _, _ = model.simulate(g, T=T, verbose=False)
        err.append(model.err[::t_sample])
        Pe_err.append(model.P_est - _P)
        _durs = []
        for k in range(len(_P)):
            ons = get_onsets(a[:,k])
            offs = get_onsets(-a[:,k])
            _durs.append(np.mean(get_durations(t, ons, offs)))
        durs.append(_durs)
    return t[::t_sample], np.mean(err, axis=0), \
           np.quantile(err, [.1, .9], axis=0), name, Pe_err, durs

def row(mdl):
    return 1 if mdl.startswith('bayes') else 0
def row_label(mdl):
    return 'Bayes' if mdl.startswith('bayes') else 'GD'
def row_label_i(i):
    return 'Bayes' if i==1 else 'GD'

def plot(res, models, axs):
    delays = list(set(r[3][1] for r in res))
    delays.sort()
    delays = dict(zip(delays, range(100)))
    ms = {}
    Ts = {}
    for lab, ax in zip(['i', 'ii'], axs):
        ax.grid(lw=.1)
        ax.text(.01, .97, f'(b{lab})', transform=ax.transAxes,
                va='top', ha='left', size='xx-small')
    for t, err, errq, (_,d,mdl), Pe_err, durs in res:
            if not mdl in ms:
                ms[mdl] = []
                Ts[mdl] = []
            Pe_err = np.mean(np.array(Pe_err)**2, axis=0)
            ms[mdl].append((d, Pe_err.mean(axis=1)))
            Ts[mdl] += durs
    for mdl, dms in ms.items():
        dms = sorted(dms, key=lambda dm: dm[0])
        ds, ms = zip(*dms)
        ax = axs[row(mdl)]
        ms = np.array(ms)
        # ax.plot(ds, ms, zorder=10, lw=.5, label=['HH', 'leisure', 'work'])
        ax.plot(ds, ms.mean(axis=1), zorder=10, lw=.75, c='k',# ls='--',
                label='avg')
        # for i, Ti in enumerate(np.array(Ts[mdl]).T):
            # ax.fill_betweenx([0, 1], *np.quantile(Ti, [.25, .75]),
                             # color=f'C{i}', lw=0, alpha=.3,
                             # transform=ax.get_xaxis_transform())
            # ax.plot([np.mean(Ti)]*2, [0, 1], c=f'C{i}', lw=.5, alpha=.5, ls=':',
                    # transform=ax.get_xaxis_transform())
        ax.plot([np.mean(Ts[mdl])]*2, [0, 1], c=f'k', lw=.5, alpha=.5, ls=':',
                transform=ax.get_xaxis_transform())
        ax.set(yscale='log')
    axs[-1].set(xlabel=r'$\Delta t$ [h]', xlim=(min(delays), max(delays)))
    # axs[-1].legend(loc='lower right', ncol=4, fontsize='xx-small',
                   # bbox_to_anchor=[0, -1.0, 1, 1], frameon=False)

def plot_trajectories(res, models, axs, cax_parent):
    delays = list(set(r[3][1] for r in res))
    cmap = plt.get_cmap('viridis')
    get_color = lambda d : cmap((d - min(delays)) / (max(delays) - min(delays)))
    for t, err, errq, (_,d,mdl), _, _ in res:
        if d not in delays[::3]: continue
        t /= 24*365
        ax = axs[row(mdl)]
        c = get_color(d)
        ax.plot(t, err, c=c, label=d, lw=.75)
        ax.fill_between(t, *errq, color=c, alpha=.2, lw=0)
    for i, ax in enumerate(axs):
        ax.set(yscale='log', ylabel=f'MSE ({row_label_i(i)})')
    axs[-1].set_xlabel('time [years]')
    for lab, ax in zip(['i', 'ii'], axs):
        ax.grid(lw=.1)
        ax.text(1-.01, 1-.03, f'(a{lab})', transform=ax.transAxes,
                va='top', ha='right', size='xx-small')
    cax = cax_parent.inset_axes([0, 1, 1, .05])
    im = cax.imshow([[]], vmin=min(delays), vmax=max(delays), aspect='auto',
                    cmap=cmap)
    plt.colorbar(im, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.set(xticks=[])

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    sd = 1
    T = 1000
    n = 10

    sigma = 0

    delays = [1, 5, 10, 25, 50]
    delays = [1, 5, 10]
    delays = np.linspace(1, 10, 5)
    dt = 1e-1
    pickle_fl = Path('data') / 'fig_delay.pickle'


    models, name, title = get_error_comparison(sigmas=[sigma],
                                               delays=delays)

    if not force and pickle_fl.exists():
        with pickle_fl.open('rb') as fl:
            res = pickle.load(fl)
    else:
        with mp.Pool(4) as p:
            res = p.map(run, models.items())
        with pickle_fl.open('wb') as fl:
            pickle.dump(res, fl)

    fig, axs = plt.subplots(2, 2, sharex='col', figsize=(halfwidth, 2))
    plot(res, models, axs[:,1])
    plot_trajectories(res, models, axs[:,0], axs[0,1])
    fig.subplots_adjust(left=half_l, top=0.965, bottom=0.2, right=0.985,
                        hspace=0.2, wspace=0.340)
    save_plot('fig_delay')
    plt.show()

