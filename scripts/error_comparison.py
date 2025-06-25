from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
import pickle
from model import Model
from setup.models import get_base_pars
from setup.model_comparison import get_error_comparison

path = Path('figures')

def run(arg):
    name, (cls, kwargs) = arg
    t_sample = kwargs['delay'] if 'delay' in kwargs else int(1/dt)
    err, Pe = [], []
    for j in range(n):
        print(f'[{j+1}/{n}] {name}', end='    \r')
        P_est = P + np.random.randn(*P.shape)*sd
        model = cls(P, G, mu, P_est=P_est, dt=dt, **kwargs)
        t, _, _, _ = model.simulate(g, T=T, verbose=False)
        err.append(model.err[::t_sample])
        # Pe.append(model.P_est)
    return t[::t_sample], err, name#, Pe

def plot(res, models, T, title, ts):
    fig, axs = plt.subplots(2, len(ts), sharex=True, sharey='row',
                            figsize=(len(ts)*3, 6))
    markers = dict(zip(set(r[2][0] for r in res), ['o', 'v', '^']))
    colors = dict(zip(set(r[2][2] for r in res), ['b', 'r']))
    delays = list(set(r[2][1] for r in res))
    delays.sort()
    delays = dict(zip(delays, range(100)))
    for t, err, (s,d,mdl) in res:
        for i, _t in enumerate(ts):
            _err = np.array(err)[:, (_t-100 < t) & (t < _t)]
            m = _err.mean(axis=1)
            std = _err.std(axis=1)
            cv = std / m
            # cv = np.abs(np.diff(_err, axis=1)).mean(axis=1) / m
            cv = np.median(np.diff(_err, axis=1), axis=1)
            cv = np.abs(np.diff(_err, axis=1).mean(axis=1))
            dd = (list(markers.keys()).index(s) + .5) / len(markers) / 5
            if mdl.startswith('grad'):
                dd *= -1
            for ax, y in zip(axs[:,i], [m, cv]):
                _m = np.median(y)
                yerr = np.abs(np.quantile(y, [.1, .9]) - _m)
                ax.errorbar([delays[d]+dd], [_m], yerr=yerr[:,None],
                            color=colors[mdl], marker=markers[s],
                            capsize=2, mfc='w')
            axs[0,i].set_title(f'after {_t-100}-{_t}h')
    axs[0,0].set(yscale='log', ylabel=r'MSE in $\Pi$')
    axs[1,0].set(yscale='log', ylabel=r'average absolute slope of error')
            #ylabel=r'CV of MSE in $\Pi$')
    for ax in axs[1,:]:
        ax.set_xticks(list(delays.values()), list(delays.keys()))
        ax.set_xlabel('delay [h]')
    legax = axs[-1,0]
    # legax.plot([], [], c='w', alpha=0, lw=0, label='models:')
    for mdl, c in colors.items():
        legax.hist([], color=c, label=mdl)
    # for _ in range(len(markers) - len(colors)):
        # legax.plot([], [], c='w', alpha=0, lw=0, label='')
    # legax.plot([], [], c='w', alpha=0, lw=0, label=r'$\sigma_u$:')
    for s, m in markers.items():
        s = s if s is not None else 0
        label = rf'$\sigma_u = {s}$'
        legax.scatter([], [], color='k', marker=m, facecolor='w',
                      label=label)
    legax.legend(ncols=2, fontsize='small', markerfirst=True, frameon=True)
    fig.suptitle(f'{title} ({n} runs)')
    fig.tight_layout()
    for ax in axs.flatten():
        ax.grid()
    return fig, axs

def plot_trajectories(res, models, title):
    rows = dict(zip(set(r[2][2] for r in res), range(100)))
    colors = dict(zip(set(r[2][1] for r in res),
                      [f'C{i}' for i in range(20)]))
    cols = dict(zip(set(r[2][0] for r in res), range(100)))
    fig, axs = plt.subplots(len(rows), len(cols), sharex=True, sharey=True)
    for t, err, (s,d,mdl) in res:
        t /= 24*365
        ax = axs[rows[mdl], cols[s]]
        # c = colors[d]
        ax.plot(t, np.median(err, axis=0), label=d)
        ax.fill_between(t, *np.quantile(err, [.1, .9], axis=0), alpha=.2, lw=0)
    for mdl, i in rows.items():
        axs[i,0].set_ylabel(f'MSE ({mdl})')
    for s, j in cols.items():
        axs[0,j].set_title(rf'$\sigma_u = {s}$')
    axs[0,-1].set(yscale='log')
    for ax in axs[-1]:
        ax.set_xlabel('time [years]')
    legax = axs[-1,-1]
    legax.legend(loc='lower right', title='delay [h]', frameon=True, ncols=2)
    for ax in axs.flatten():
        ax.grid()
    return fig, axs

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    models, name, title = get_error_comparison(sigmas=[None, 3],
                                               # delays=[5, 50])
                                               delays=[5, 10, 25, 50])

    sd = 1
    T = 25000
    n = 15
    dt = 5e-1

    pickle_fl = Path('data') / 'error_comparison_runs.pickle'
    """
    with mp.Pool(4) as p:
        res = p.map(run, models.items())
    with pickle_fl.open('wb') as fl:
        pickle.dump(res, fl)
    """
    with pickle_fl.open('rb') as fl:
        res = pickle.load(fl)
    # """

    # fig, axs = plot(res, models, T, title, ts=[500, T//2, T])
    # fig.savefig(path / f'{name}.png')
    fig, axs = plot_trajectories(res, models, title)
    fig.savefig(path / f'{name}_trajectories.png')
    plt.show()

