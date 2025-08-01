from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
import pickle
from model import Model
from setup.models import get_base_pars
from setup.model_comparison import get_learning_basin_comparison

path = Path('figures')

def run(arg):
    name, (cls, kwargs) = arg
    t_sample = kwargs['delay'] if 'delay' in kwargs else int(1/dt)
    err, Pe, ds, Pe0 = [], [], [], []
    for j in range(n):
        print(f'[{j+1}/{n}] {name}', end='    \r')
        P_est, d = sample_P0()
        model = cls(P, G, mu, P_est=P_est, dt=dt, **kwargs)
        t, _, _, _ = model.simulate(g, T=T, verbose=False)
        err.append(model.err[::t_sample])
        Pe.append(model.P_est)
        Pe0.append(P_est)
        ds.append(d)
    return t[::t_sample], err, name, Pe, Pe0, ds

def plot(res):
    fig, ax = plt.subplots()
    markers = dict(zip(set(r[2][0] for r in res), ['o', 'v', '^']))
    ls = dict(zip(set(r[2][0] for r in res), ['-', '--', ':']))
    colors = dict(zip(set(r[2][1] for r in res), ['b', 'r']))
    print(P)

    l, k = P.shape
    fig2, axs = plt.subplots(l, k, sharey=True, figsize=(10,8))
    for t, err, (sd, m), Pe, Pe0, ds in res:
        # if m.startswith('bayes'): continue
        # if sd is not None: continue
        err = np.array(err)
        ds = np.array(ds)
        # err /= err[:,:1]
        e = err[:,-1]#-err[:,:10].mean(axis=1)
        e[e>1e10] = np.nan
        ax.scatter(np.abs(ds), e, marker=markers[sd],
                   color=colors[m], label=rf'{m}--$\sigma_u = {sd}$')
        ax.vlines(np.abs(ds[np.isnan(e)]), 0, 1, color=colors[m],
                  transform=ax.get_xaxis_transform(), ls=ls[sd])
        # mask =  > 5
        # print(np.sum(mask))
        # ax.plot(t, err[mask].T, lw=1)
        if m.startswith('bayes'): continue
        Pe0 = np.array(Pe0)
        for i in range(l):
            for j in range(k):
                pij0 = Pe0[:,i,j]
                _ax = axs[i,j]
                bins = np.linspace(-10, 10, 10)
                _, _, patches = _ax.hist(pij0, histtype='step', bins=bins)
                _ax.hist(pij0[np.isnan(e)], alpha=.3, bins=bins,
                         color=patches[0].get_edgecolor())
    for i in range(l):
        for j in range(k):
            _ax = axs[i,j]
            _ax.vlines(P[i,j], 0, 1, color='k',
                    transform=_ax.get_xaxis_transform(), ls='-')
    ax.plot([0,1], [1,1], transform=ax.get_yaxis_transform(), c='k', ls='--')
    ax.legend()
    ax.set(yscale='log', xlabel=r'$| \hat{\Pi}_0 - \Pi |$', ylabel='error')


def sample_P0():
    d = np.random.uniform(-sd/2, sd/2)
    P0 = np.random.randn(*P.shape)
    P0 /= np.linalg.norm(P0, 'fro')
    d += np.sign(d) * 5
    return d * P0 + P, d

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    models, name, title = get_learning_basin_comparison(sigmas=[None, 2],
                                                        delay=5, eta=1e-3)
    models = {(sd,m): v for (sd,m), v in models.items()
                        if not m.startswith('bayes')}

    sd = 25
    T = 3000
    n = 80
    dt = 5e-1


    pickle_fl = Path('data') / 'learning_basin2.pickle'
    """
    with mp.Pool(4) as p:
        res = p.map(run, models.items())
    with pickle_fl.open('wb') as fl:
        pickle.dump(res, fl)
    """
    with pickle_fl.open('rb') as fl:
        res = pickle.load(fl)
    # """
    
    plot(res)
    plt.show()
