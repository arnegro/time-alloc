from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
import pickle
from model import Model
from setup.models import get_base_pars, get_feedback_model_pars
from setup.model_comparison \
            import get_error_comparison, get_error_comparison_du, get_error_comparison_multitask, get_error_comparison_fig_multitask
from plotting import colwidth, save_plot

path = Path('figures')
force = True

def run(arg):
    name, (cls, kwargs) = arg
    t_sample = int(kwargs['delay']/dt) if 'delay' in kwargs else int(1/dt)
    err, Pe = [], []
    default_kwargs = dict(P=P, G=G, mu=mu, dt=dt)
    kwargs = default_kwargs | kwargs
    _P = kwargs['P']
    for j in range(n):
        print(f'[{j+1}/{n}] {name}', end='    \r')
        P_est = _P + np.random.randn(*_P.shape)*sd
        model = cls(P_est=P_est, **kwargs)
        t, a, _, _ = model.simulate(g, T=T, verbose=False)
        err.append(model.err[::t_sample])
        # Pe.append(model.P_est)
    print('----')
    print(model.G)
    print(model.P)
    print(model.mu)
    print(model)
    print(f'[{j+1}/{n}] {name} {np.mean(err, axis=0)[-1]}', end='    \n')
    print('----')
    return t[::t_sample], np.median(err, axis=0), \
           np.quantile(err, [.1, .9], axis=0), name#, Pe

def plot(res, models, title):
    markers = dict(zip(set(r[3][0] for r in res), ['o', 'v', '^']))
    axes = dict(zip(set(r[3][0] for r in res), range(100)))
    colors = dict(zip(set(r[3][2] for r in res), ['b', 'r', 'g', 'magenta']))
    delays = list(set(r[3][1] for r in res))
    fig, axs = plt.subplots(1, len(axes), sharex=True, sharey='row',
                            figsize=(colwidth, 2))
    if len(axes) == 1:
        axs = [axs]
    delays.sort()
    delays = dict(zip(delays, range(100)))
    ms = {}
    for ax in axs:
        ax.grid()
    for t, err, errq, (s,d,mdl) in res:
            if not (s,mdl) in ms:
                ms[(s,mdl)] = []
            _err = np.array(err)[t.max()-100 < t]
            _errq = np.array(errq)[:, t.max()-100 < t]
            m = _err.mean()
            # dd = (list(markers.keys()).index(s) + .5) / len(markers) / 1
            # dd = .2
            dd = 0
            # if mdl.startswith('grad'):
                # dd *= -1
            yerr = np.abs(np.mean(_errq, axis=1) - m)
            ax = axs[axes[s]]
            ax.errorbar([d+dd], [m], yerr=yerr[:,None],
                        color=colors[mdl], marker=markers[s],
                        capsize=2, mfc='w', ms=1.5**2, zorder=100, lw=.5, mew=.5)
            # ax.set_title(f'after {_t-100}-{_t}h', size='xx-small')
            ms[(s,mdl)].append((d,m))
    for (s,mdl), dms in ms.items():
        dms = sorted(dms, key=lambda dm: dm[0])
        ds, ms = zip(*dms)
        # dd = (list(markers.keys()).index(s) + .5) / len(markers) / 1
        # dd = .2
        # if mdl.startswith('grad'):
            # dd *= -1
        ax = axs[axes[s]]
        ax.plot(np.array(ds)+dd, ms, color=colors[mdl], zorder=10, lw=.75)
    axs[0].set(yscale='log', ylabel=r'MSE in $\Pi$')
    for ax in axs:
        ax.set_xlabel('delay [h]')
    for mdl, c in colors.items():
        axs[0].plot([], color=c, label=mdl)
    for s, m in markers.items():
        s = s if s is not None else 0
        label = rf'$\sigma_u = {s}$'
        axs[0].scatter([], [], color='k', marker=m, facecolor='w', s=3**2,
                       lw=.5, label=label)
    fig.legend(*axs[0].get_legend_handles_labels(), ncols=4,
               loc='lower right',
               fontsize='xx-small', markerfirst=True, frameon=False)
    fig.subplots_adjust(top=0.985, bottom=0.286, left=0.166, right=0.99,
                        hspace=0.2, wspace=0.147)
    return fig, axs

def plot_trajectories(res, models, title):
    rows = dict(zip(set(r[3][2] for r in res), range(100)))
    colors = dict(zip(set(r[3][1] for r in res),
                      [f'C{i}' for i in range(20)]))
    cols = dict(zip(set(r[3][0] for r in res), range(100)))
    delays = list(set(r[3][1] for r in res))
    cmap = plt.get_cmap('viridis')
    get_color = lambda d : cmap((d - min(delays)) / (max(delays) - min(delays)))
    fig, axs = plt.subplots(len(rows), len(cols), sharex=True, sharey='row',
                            figsize=(colwidth, 2))
    if len(cols) == 1 and len(rows) == 1:
        axs = np.array([[axs]])
    elif len(cols) == 1:
        axs = axs[:,None]
    elif len(rows) == 1:
        axs = axs[None,:]
    axs[-1,-1].plot([], [], lw=0, c='w', label=r'$\Delta t$ [h]:')
    for t, err, errq, (s,d,mdl) in res:
        t /= 24*365
        ax = axs[rows[mdl], cols[s]]
        c = get_color(d)#colors[d]
        ax.plot(t, err, c=c, label=d, lw=.75)
        ax.fill_between(t, *errq, color=c, alpha=.2, lw=0)
    for mdl, i in rows.items():
        axs[i,0].set_ylabel(f'MSE\n({mdl})')
        axs[i,0].set(yscale='log')
    for s, j in cols.items():
        axs[0,j].set_title(rf'$\sigma_u = {s}$', size='x-small')
    for ax in axs[-1]:
        ax.set_xlabel('time [years]')
    # for legax in axs[:,-1]:
        # legax.legend(title='delay [h]', frameon=True, ncols=2)
    # fig.legend(*axs[-1,-1].get_legend_handles_labels(), loc='lower center',
               # frameon=False, fontsize='xx-small', ncols=len(colors)+1)#,
               # title=r'$\Delta t$ [h]')
    for ax in axs.flatten():
        ax.grid()
    left, right = .204, 0.983
    fig.subplots_adjust(top=0.909, bottom=0.306, left=left, right=right,
                        hspace=0.2, wspace=0.132)
    cax = fig.add_subplot([left, .09, right-left, .01])
    im = cax.imshow([[]], vmin=min(delays), vmax=max(delays), aspect='auto',
                    cmap=cmap)
    cax.text(-.02, .5, r'$\Delta t$ [h]:', ha='right', va='center',
             transform=cax.transAxes, size='xx-small')
    plt.colorbar(im, cax=cax, orientation='horizontal')
    return fig, axs

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    sd = 1
    T = 5000
    n = 2

    sigmas = [None, 2]
    # sigmas = [None]

    delays = [1, 5, 10, 25, 50]
    dt = 1e-1
    pickle_fl = Path('data') / 'error_comparison_runs.pickle'
    name_ext = '-base_model'

    # delays = [.1, 1, 5]
    # dt = 5e-3
    # pickle_fl = Path('data') / 'error_comparison_runs2.pickle'

    # delays = [.1, 1, 5]
    # eta = 1e-3
    # dt = 1e-1
    # pickle_fl = Path('data') / 'error_comparison_runs3.pickle'

    P, G, mu, g = get_feedback_model_pars(g12=5, mu3=20)
    delays = [1, 2, 5, 10, 15, 20]
    delays = [1, 5, 20]
    dt = 1e-1
    pickle_fl = Path('data') / 'error_comparison_runs4.pickle'
    name_ext = '-feedback_model'

    models, name, title = get_error_comparison(sigmas=sigmas,
                                               delays=delays)

    models, name, title = get_error_comparison_multitask(delays=[5])
    models, name, title = get_error_comparison_fig_multitask(gammas=[0, 10/3],
                                                             sigma_u=None,
                                                             delays=[5])
    name_ext = '-multitask'
    pickle_fl = Path('data') / 'error_comparison_runs7.pickle'
    """
    models, name, title = get_error_comparison_du(sigmas=sigmas, dt=1e-2,
                                                  # dts=[1e-1, 1e-2, 1e-3],
                                                  delays=delays, eta=eta)
    """
    name += name_ext
    # models = {(s,d,m): v for (s,d,m), v in models.items()
                         # if m.startswith('grad')}

    # """
    if not force and pickle_fl.exists():
        with pickle_fl.open('rb') as fl:
            res = pickle.load(fl)
    else:
        with mp.Pool(4) as p:
            res = p.map(run, models.items())
        with pickle_fl.open('wb') as fl:
            pickle.dump(res, fl)
    quit()

    fig, axs = plot(res, models, title)
    save_plot(name)
    fig, axs = plot_trajectories(res, models, title)
    save_plot(name+'_trajectories')
    plt.show()
