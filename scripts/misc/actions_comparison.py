from pathlib import Path
import pickle
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
from model import Model
from setup.models import get_base_pars
from setup.model_comparison import get_action_comparison

path = Path('figures')
force = False

def run(arg):
    name, (cls, kwargs) = arg
    t_sample = kwargs['delay'] if 'delay' in kwargs else int(1/dt)
    if cls != Model:
        err, a = [], []
        for j in range(n):
            print(f'[{j+1}/{n}] {name}', end='    \r')
            P_est = P + np.random.randn(*P.shape)*sd
            model = cls(P, G, mu, P_est=P_est, dt=dt, **kwargs)
            t, _a, u, u_est = model.simulate(g, T=T, verbose=False)
            a.append(_a)
            err.append(model.err[::t_sample])
    else:
        model = cls(P, G, mu, dt=dt, **kwargs)
        t, a, u = model.simulate(g, T=T, verbose=False)
        err = None
        a = [a]
    ta = t.copy()
    t = t[::t_sample]
    return t, ta, a, err, name

def get_onsets(a):
    return np.where(np.diff((a > 0).astype(int), axis=0) > .5)[0]

def smooth(x, dt, n=7):
    n = int(n/dt)
    ker = np.ones(n) / n
    n *= 5
    _x = np.concatenate([[x[0]]*n, x, [x[-1]]*n])
    return np.convolve(_x, ker, mode='same')[n:-n]

def get_IEIs(ta, ons, burnin=20):
    def y(_ons):
        _t = ta[_ons]
        _t = _t[_t > burnin]
        return np.diff(_t)
    ys = [y(_ons)*24 for _ons in ons]
    return ys

def get_CVs(ta, ons):
    dt = np.diff(ta).mean()
    y = lambda _ons : np.interp(ta, ta[_ons[:-1]], np.diff(ta[_ons]))
    ys = [y(_ons) for _ons in ons]
    ms = [smooth(_ys, dt) for _ys in ys]
    CVs = [np.sqrt(smooth((_ys - _ms)**2, dt))/_ms
                          for _ys, _ms in zip(ys, ms)]
    return CVs

def get_durations(ta, ons, offs):
    dt = np.diff(ta).mean()
    ds = []
    for _ons, _offs in zip(ons, offs):
        if _offs[0] < _ons[0]:
            _offs = _offs[1:]
        _ons = _ons[:len(_offs)]
        ds.append(smooth(np.interp(ta, ta[_ons], ta[_offs] - ta[_ons]), dt)*24)
    return ds

def get_time_per_day(ta, a):
    dt = np.diff(ta).mean() * 24
    a = (a > 0).astype(int) / dt
    return smooth(smooth(a, dt, n=24) * dt * 24, dt/24)

def plot(res, models, T, title, k, qs=[.25, .75]):
    models = list(set(_k[1] for _k in models.keys()))
    fig, axs = plt.subplots(len(sigmas), 5, sharex='col', sharey='col',
                            figsize=(10,5))
    bins = np.geomspace(1, 48, 24)
    for t, ta, a, err, (s, m) in res:
        print(s, m)
        t = t/24
        ta = ta/24
        i = sigmas.index(s)
        j = models.index(m)
        c = f'C{j}'
        ons = [get_onsets(_a[:,k]) for _a in a]
        offs = [get_onsets(-_a[:,k]) for _a in a]
        CVs = get_CVs(ta, ons)
        ds = get_durations(ta, ons, offs)
        time_per_day = [get_time_per_day(ta, _a[:,k]) for _a in a]
        IEIs = get_IEIs(ta, ons)
        IEIs = [IEI for _IEIs in IEIs for IEI in _IEIs]
        if m != 'basic model':
            for l, y in enumerate([CVs, ds, time_per_day]):
                axs[i, l].plot(ta, np.median(y, axis=0), c=c)
                axs[i, l].fill_between(ta, *np.quantile(y, qs, axis=0),
                                       alpha=.2, lw=0, color=c)
            axs[i, -2].hist(IEIs, bins=bins, histtype='step', color=c)
            axs[i, -1].plot(t, np.median(err, axis=0), c=c, label=m)
            axs[i, -1].fill_between(t, *np.quantile(err, qs, axis=0),
                                    alpha=.2, lw=0, color=c)
        else:
            for l, y in enumerate([CVs, ds, time_per_day]):
                y = y[0][ta > 20]
                axs[i, l].plot(ta, [np.median(y)]*len(ta), c='k', ls='--',
                                zorder=-10)
                q1, q2 = np.quantile(y, qs)
                axs[i, l].fill_between(ta, [q1]*len(ta), [q2]*len(ta),
                                        alpha=.2, lw=0, color='k',
                                        zorder=-10)
            _,_,p = axs[i, -2].hist(IEIs*n, bins=bins, color='k',
                                    histtype='step', linestyle='--',
                                    zorder=100)
            p[0].set_linewidth(2)
    for i, s in enumerate(sigmas):
        axs[i, 0].set_ylabel(rf'$\sigma_u = {s}$')
    y0, y1 = axs[0,0].get_ylim()
    axs[0, 0].set(yscale='log', title='CV of IEI', ylim=(max(1e-3, y0), y1))
    axs[0, 1].set(yscale='log', title='durations [h]')
    axs[0, 2].set(yscale='linear', title='time per day [h]')
    axs[0, -2].set(xscale='log', yscale='log', title='IEI distribution')
    axs[0, -1].set(yscale='log', title=r'error in $\Pi$')
    for _ax in axs[-1, np.arange(axs.shape[1])!=3]:
        _ax.set(xlabel='time [days]', xlim=(0, ta.max()-5))
    axs[-1, -2].set_xlabel('IEI [h]')
    axs[-1,-1].legend(frameon=False)
    return fig, axs

def base_model_summary(res, models):
    fig, axs = plt.subplots(3, 1, sharex=True)
    lines = {(j,k): [] for j in range(len(axs)) for k in range(len(mu))}
    for t, ta, a, err, (s, m) in res:
        ta = ta/24
        if not m.startswith('basic'):
            continue
        if s is None:
            s = 0
        for k in range(len(mu)):
            ons = [get_onsets(_a[:,k]) for _a in a]
            offs = [get_onsets(-_a[:,k]) for _a in a]
            CVs = get_CVs(ta, ons)
            ds = get_durations(ta, ons, offs)
            time_per_day = [get_time_per_day(ta, _a[:,k]) for _a in a]
            for j, y in enumerate([CVs, ds, time_per_day]):
                m = np.mean(y)
                q = np.quantile(y, [.25, .75])
                yerr = np.abs(q - m)
                dx = (k - np.mean(range(len(actions)))) * .1
                axs[j].errorbar([s+dx], [m], yerr=yerr[:,None], capsize=2,
                                color=f'C{k}', marker='o')
                lines[(j,k)].append((s+dx, m))
    for (j,k), line in lines.items():
        axs[j].plot(*zip(*line), c=f'C{k}', zorder=-10, label=actions[k])
    axs[0].set_ylabel('CV of IEIs')
    axs[1].set_ylabel('duration [h]')
    axs[2].set_ylabel('time per day [h]')
    axs[0].legend(ncols=3, frameon=False)
    axs[-1].set_xlabel(r'$\sigma_u$')
    for ax in axs:
        ax.grid()
    return fig, axs



if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    sigmas = [None, 1, 3, 5]
    # sigmas = [None, 5]
    models, name, title = get_action_comparison(sigmas=sigmas)

    actions = ['HH', 'leisure', 'work']

    sd = 1
    T = 8000
    n = 10
    dt = 1e-1

    pickle_fl = Path('data') / f'{name}.pickle'
    if not force and pickle_fl.exists():
        with pickle_fl.open('rb') as fl:
            res = pickle.load(fl)
    else:
        with mp.Pool(4) as p:
            res = p.map(run, models.items())
        with pickle_fl.open('wb') as fl:
            pickle.dump(res, fl)

    fig, axs = base_model_summary(res, models)
    fig.suptitle(f'{title}\nComparison of stats for base model')
    fig.tight_layout()
    fig.savefig(path / f'{name}-base.png')
    plt.show()
    quit()

    for k in range(len(mu)):
        fig, axs = plot(res, models, T, title, k)
        alab = actions[k]
        fig.suptitle(f'{title}---{alab}')
        fig.tight_layout()
        fig.savefig(path / f'{name}-{alab}.png')
        plt.show()
        plt.close()
