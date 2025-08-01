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
    fig, axs = plt.subplots(len(ts), 1, sharex=True, figsize=(10,6))
    markers = dict(zip(set(r[2][2] for r in res), ['o', 'v', '^']))
    colors = dict(zip(set(r[2][2] for r in res), ['b', 'r']))
    errs = {i: {} for i in range(len(ts))}
    for t, err, (s,_,mdl) in res:
        for i, _t in enumerate(ts):
            if mdl not in errs[i]:
                errs[i][mdl] = {}
                label = mdl
            else:
                label = None
            _err = np.array(err)[:, (_t-100 < t) & (t < _t)]
            m = _err.mean(axis=1)
            q = np.quantile(m, [.1, .9])
            m = np.median(m)
            yerr = np.abs(q - m)
            errs[i][mdl][s] = m
            axs[i].errorbar([s], [m], yerr=yerr[:,None], label=label,
                            color=colors[mdl], marker=markers[mdl], capsize=2)
            if _t < 24*365:
                _title = rf'after $\sim$ {int(_t/24)} days'
            else:
                _title = rf'after $\sim$ {int(_t/24/365)} years'
            axs[i].set_title(_title)
    for i, _ in enumerate(ts):
        _errs = errs[i]
        for mdl, __errs in _errs.items():
            _s, _err = zip(*sorted(__errs.items()))
            axs[i].plot(_s, _err, color=colors[mdl], zorder=-20)
        axs[i].set(yscale='log', ylabel=r'MSE in $\Pi$')
    axs[-1].set_xlabel(r'$\sigma_u$')
    legax = axs[-1]
    legax.legend(ncols=1, markerfirst=True, frameon=True)
    fig.suptitle(f'{title} ({n} runs)')
    fig.tight_layout()
    for ax in axs:
        ax.grid()
    return fig, axs

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    sigmas = np.concatenate([[0], np.geomspace(0.01, 3, 10)])
    models, name, title = get_error_comparison(sigmas=sigmas, delays=[25])
    name = 'noise_comparison'

    sd = 1
    T = 20000
    n = 15
    dt = 5e-1

    pickle_fl = Path('data') / 'noise_comparison_runs.pickle'
    """
    with mp.Pool(4) as p:
        res = p.map(run, models.items())
    with pickle_fl.open('wb') as fl:
        pickle.dump(res, fl)
    """
    with pickle_fl.open('rb') as fl:
        res = pickle.load(fl)
    # """

    fig, axs = plot(res, models, T, title, ts=[500, T])
    fig.savefig(path / f'{name}.png')
    plt.show()

