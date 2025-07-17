from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import multiprocessing as mp
import pickle
from model import Model
from setup.models import get_base_pars, get_feedback_model_pars
from setup.model_comparison import get_error_comparison_fig_multitask
from plotting import halfwidth, save_plot, half_l

path = Path('figures')
force = False

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
    return t[::t_sample], np.mean(err, axis=0), \
           np.quantile(err, [.1, .9], axis=0), name

def row(mdl):
    return 1 if mdl.startswith('bayes') else 0
def row_label(mdl):
    return 'Bayes' if mdl.startswith('bayes') else 'GD'
def row_label_i(i):
    return 'Bayes' if i==1 else 'GD'
def model_color(mdl):
    return 'g' if mdl.startswith('bayes') else 'b'

def plot(res, models, axs):
    gammas = list(set(r[3][0] for r in res))
    delays = list(set(r[3][1] for r in res))
    delays.sort()
    ls = dict(zip(delays, [':', '-', '--']))
    colors = dict(zip(delays, ['g', 'r', 'b']))
    ms = {}
    for lab, ax in zip(['i', 'ii'], axs):
        ax.grid(lw=.1)
        ax.text(.01, .97, f'(b{lab})', transform=ax.transAxes,
                va='top', ha='left', size='xx-small')
    for t, err, errq, (gamma,d,mdl) in res:
            if not mdl in ms:
                ms[mdl] = {}
            if not d in ms[mdl]:
                ms[mdl][d] = []
            ms[mdl][d].append((gamma, err[-1]))
    for mdl, dms in ms.items():
        for d, gms in dms.items():
            dms = sorted(gms, key=lambda gm: gm[0])
            gs, ms = zip(*gms)
            ax = axs[row(mdl)]
            ax.plot(gs, ms, zorder=10, lw=.75, c=colors[d], ls=ls[d])
            ax.set(yscale='log')
    axs[-1].set(xlabel=r'$\gamma$', xlim=(min(gammas), max(gammas)))

def plot_trajectories(res, models, axs, cax_parent):
    gammas = list(set(r[3][0] for r in res))
    sorted(gammas)
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=min(gammas), vmax=max(gammas))
    get_color = lambda s : cmap(norm(s))
    for t, err, errq, (gamma,d,mdl) in res:
        if d != 5: continue
        t /= 24*365
        ax = axs[row(mdl)]
        c = get_color(gamma)
        # if s not in sigmas[::2]: continue
        ax.plot(t, err, c=c, lw=.75)
        ax.fill_between(t, *errq, color=c, alpha=.2, lw=0)
    for i, ax in enumerate(axs):
        ax.set(yscale='log', ylabel=f'MSE ({row_label_i(i)})')
    axs[-1].set_xlabel('time [years]')
    for lab, ax in zip(['i', 'ii'], axs):
        ax.grid(lw=.1)
        ax.text(.01, .03, f'(a{lab})', transform=ax.transAxes,
                va='bottom', ha='left', size='xx-small')
    cax = cax_parent.inset_axes([0, 1, 1, .04])
    im = cax.imshow([[]], vmin=min(gammas), vmax=max(gammas), aspect='auto',
                    cmap=cmap)
    plt.colorbar(im, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.set(xticks=[])

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    sd = 1
    T = 2000
    n = 5

    sigma = None

    gammas = np.linspace(-1, 3, 6)
    delays = [2, 5, 10]
    dt = 1e-1
    pickle_fl = Path('data') / 'fig_multitask.pickle'


    models, name, title = get_error_comparison_fig_multitask(gammas=gammas,
                                                             sigma_u=sigma,
                                                             delays=delays)

    if not force and pickle_fl.exists():
        with pickle_fl.open('rb') as fl:
            res = pickle.load(fl)
    else:
        with mp.Pool(4) as p:
            res = p.map(run, models.items())
        with pickle_fl.open('wb') as fl:
            pickle.dump(res, fl)

    fig, axs = plt.subplots(2, 2, sharex='col',# sharey='col',
                            figsize=(halfwidth, 2))
    plot(res, models, axs[:,1])
    plot_trajectories(res, models, axs[:,0], axs[0,1])
    fig.subplots_adjust(left=half_l, top=0.965, bottom=0.2, right=0.985,
                        hspace=0.2, wspace=0.340)
    save_plot('fig_multitask')
    plt.show()

