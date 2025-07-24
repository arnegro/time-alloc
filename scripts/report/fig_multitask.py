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
    default_kwargs = dict(P=P, G=G, mu=mu, dt=dt)
    kwargs = default_kwargs | kwargs
    _P = kwargs['P']
    corr = []
    for j in range(n):
        print(f'[{j+1}/{n}] {name}', end='    \r')
        P_est = _P + np.random.randn(*_P.shape)*sd
        model = cls(P_est=P_est, **kwargs)
        t, a, _, _ = model.simulate(g, T=T, verbose=False)
        err.append(model.err[::t_sample])
        _a = np.clip(a, a_min=0, a_max=None)
        corr.append(np.corrcoef(_a[:,:2], rowvar=False)[1,0])
    return t[::t_sample], np.mean(err, axis=0), \
           np.quantile(err, [.1, .9], axis=0), np.mean(corr), name

def row(mdl):
    return 1 if mdl.startswith('bayes') else 0
def row_label(mdl):
    return 'Bayes' if mdl.startswith('bayes') else 'GD'
def row_label_i(i):
    return 'Bayes' if i==1 else 'GD'
def model_color(mdl):
    return 'g' if mdl.startswith('bayes') else 'b'

def plot(res, models, axs):
    gammas = list(set(r[4][0] for r in res))
    delays = list(set(r[4][1] for r in res))
    delays.sort()
    delays.remove(5)
    ls = dict(zip(delays, [':', '--']))
    ms = {}
    for lab, ax in zip(['i', 'ii'], axs):
        ax.grid(lw=.1)
        if lab == 'i':
            bbox = dict(facecolor='w', pad=.5, lw=0, alpha=1)
        else:
            bbox = None
        ax.text(.01, .97, f'(b{lab})', transform=ax.transAxes,
                va='top', ha='left', size='xx-small', bbox=bbox, zorder=1000)
    for t, err, errq, corr, (gamma,d,mdl) in res:
            if not mdl in ms:
                ms[mdl] = {}
            if not d in ms[mdl]:
                ms[mdl][d] = []
            ms[mdl][d].append((gamma, err[-1], corr))
    axs[0].plot([], [], lw=0, label=r'$\Delta t$ [h]:')
    axs[-1].set(xlabel=r'$\gamma$', xlim=(min(gammas), max(gammas)))
    for mdl, dms in ms.items():
        ax = axs[row(mdl)]
        if mdl.startswith('grad'):
            _ax = ax.inset_axes([0.04,0.08,.3,.3])
            _ax.text(.02, .04, '(c)', size=6, ha='left', va='bottom',
                     transform=_ax.transAxes)
        for d, gms in dms.items():
            dms = sorted(gms, key=lambda gm: gm[0])
            gs, ms, cs = zip(*gms)
            kwargs = dict(ls='-', c='k')
            if d != 5:
                kwargs['ls'] = ls[d]
                kwargs['lw'] = .5
            ax.plot(gs, ms, zorder=10, label=d, **kwargs)
            if d == 5 and mdl.startswith('grad'):
                _ax.plot(gs, cs, c='k', lw=.3)
                # _ax.imshow([cs], aspect='auto', cmap='seismic', vmin=-1, vmax=1)
        ax.set(yscale='log')
        if mdl.startswith('grad'):
            _ax.set(xticklabels=[], xticks=ax.get_xticks(),# yticks=[],
                    xlim=(min(gammas), max(gammas)))
            _ax.set_title('corr HH/leisure', size=4, y=.8, ha='left', x=0)
            # _ax.xaxis.tick_top()
            _ax.xaxis.set_label_position('top')
            _ax.yaxis.tick_right()
            _ax.yaxis.set_label_position('right')
            _ax.tick_params(labelsize=4, size=2)
    axs[0].legend(fontsize=6, ncols=4, frameon=False,
                  borderpad=0, borderaxespad=0,
                  bbox_to_anchor=[0, 1.15, 1, 1], loc='lower right')
    y0, y1 = axs[0].get_ylim()
    axs[0].set_ylim(y0, y1*100)

def plot_trajectories(res, models, axs, cax_parent):
    gammas = list(set(r[4][0] for r in res))
    sorted(gammas)
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=min(gammas), vmax=max(gammas))
    get_color = lambda s : cmap(norm(s))
    for t, err, errq, _, (gamma,d,mdl) in res:
        if d != 5: continue
        t /= 24*365
        ax = axs[row(mdl)]
        c = get_color(gamma)
        if gamma not in gammas[::2]: continue
        ax.plot(t, err, c=c, lw=.75)
        ax.fill_between(t, *errq, color=c, alpha=.2, lw=0)
    for i, ax in enumerate(axs):
        ax.set(yscale='log', ylabel=f'MSE ({row_label_i(i)})')
    axs[-1].set_xlabel('time [years]')
    for lab, ax in zip(['i', 'ii'], axs):
        ax.grid(lw=.1)
        ax.text(1-.01, 1-.03, f'(a{lab})', transform=ax.transAxes,
                va='top', ha='right', size='xx-small')
    cax = cax_parent.inset_axes([0, 1, 1, .04])
    im = cax.imshow([[]], vmin=min(gammas), vmax=max(gammas), aspect='auto',
                    cmap=cmap)
    plt.colorbar(im, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.set(xticks=[])

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    sd = 1
    T = 10000
    n = 5

    sigma = None

    gammas = np.linspace(-1, 3, 10)
    delays = [2, 5, 10]
    dt = 1e-1
    pickle_fl = Path('data') / 'fig_multitask.pickle'


    _, _, _, g = get_feedback_model_pars(g12=5, mu3=20)
    models, name, title = get_error_comparison_fig_multitask(gammas=gammas,
                                                             sigma_u=sigma,
                                                             delays=delays)
    # print(models)
    # models, name, title = get_error_comparison_multitask(delays=[5, 6])
    # print(models)
    # quit()

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
    fig.subplots_adjust(left=half_l, top=0.905, bottom=0.17, right=0.985,
                        hspace=0.2, wspace=0.340)
    fig.align_ylabels()
    save_plot('fig_multitask')
    plt.show()

