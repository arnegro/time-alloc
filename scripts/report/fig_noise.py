from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import multiprocessing as mp
import pickle
from model import Model
from setup.models import get_base_pars, get_feedback_model_pars
from setup.model_comparison \
            import get_error_comparison, get_error_comparison_du, get_error_comparison_multitask
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

def plot(res, models, axs):
    sigmas = list(set(r[3][0] for r in res))
    ms = {}
    ax.grid(lw=.1)
    for t, err, errq, (s,_,mdl) in res:
        if s == 0:
            ax.scatter([0], [err[-1]], transform=ax.get_yaxis_transform(),
                       clip_on=False, zorder=1000)
            continue
        if not mdl in ms:
            ms[mdl] = []
        ms[mdl].append((s, err[-1]))
    for mdl, sms in ms.items():
        sms = sorted(sms, key=lambda sm: sm[0])
        ss, ms = zip(*sms)
        ax.plot(ss, ms, zorder=10, lw=.75,# c='k',# ls='--',
                label='avg')
    sigmas.pop(0)
    ax.set(xlabel=r'$\sigma_u$', xlim=(min(sigmas), max(sigmas)), 
           xscale='log', yscale='log')

def plot_trajectories(res, models, axs, cax_parent):
    sigmas = list(set(r[3][0] for r in res))
    sigmas.pop(0)
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=min(sigmas), vmax=max(sigmas))
    get_color = lambda s : cmap(norm(s))
    for t, err, errq, (s,_,mdl) in res:
        # if s not in sigmas[::3]: continue
        t /= 24*365
        ax = axs[row(mdl)]
        if s == 0:
            c = 'r'
        else:
            c = get_color(s)
        ax.plot(t, err, c=c, lw=.75)
        ax.fill_between(t, *errq, color=c, alpha=.2, lw=0)
    for i, ax in enumerate(axs):
        ax.set(yscale='log', ylabel=f'MSE ({row_label_i(i)})')
    axs[-1].set_xlabel('time [years]')
    for ax in axs.flatten():
        ax.grid(lw=.1)
    cax = cax_parent.inset_axes([0, 1, 1, .02])
    im = cax.imshow([[]], vmin=min(sigmas), vmax=max(sigmas), aspect='auto',
                    cmap=cmap, norm='log')
    plt.colorbar(im, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.set(xticks=[])

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    sd = 1
    T = 2000
    n = 5

    sigma = 0

    sigmas = np.concatenate([[0], np.geomspace(1e-4, 3, 3)])
    dt = 1e-1
    pickle_fl = Path('data') / 'fig_noise.pickle'


    models, name, title = get_error_comparison(sigmas=sigmas,
                                               delays=[5])

    if not force and pickle_fl.exists():
        with pickle_fl.open('rb') as fl:
            res = pickle.load(fl)
    else:
        with mp.Pool(4) as p:
            res = p.map(run, models.items())
        with pickle_fl.open('wb') as fl:
            pickle.dump(res, fl)

    fig, axs = plt.subplots(2, 2, sharex='col', figsize=(halfwidth, 2))
    gs = axs[0,0].get_gridspec()
    for ax in axs[:,1]:
        ax.remove()
    ax = fig.add_subplot(gs[:,1])
    plot(res, models, ax)
    plot_trajectories(res, models, axs[:,0], ax)
    fig.subplots_adjust(left=half_l, top=0.965, bottom=0.2, right=0.985,
                        hspace=0.2, wspace=0.340)
    save_plot('fig_delay')
    plt.show()

