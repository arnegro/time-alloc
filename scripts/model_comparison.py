from pathlib import Path
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
from model import Model
from setup.models import get_base_pars, get_feedback_model_pars
from setup.model_comparison \
        import get_base_model_comparison, get_delay_comparison, \
               get_noise_comparison, compare_prob, get_wrong_P_comparison, \
               get_g_bias_comparison

path = Path('figures')
force = True

def run(arg):
    name, (cls, kwargs) = arg
    default_kwargs = dict(P=P, G=G, mu=mu, dt=dt)
    kwargs = default_kwargs | kwargs
    _dt = kwargs['dt']
    t_sample = int(kwargs['delay']/_dt) if 'delay' in kwargs else int(1/_dt)
    if cls != Model:
        m = np.random.randint(n)
        err, Pe = [], []
        for j in range(n):
            print(f'[{j+1}/{n}] {name}', end='    \r')
            _kwargs = kwargs.copy()
            if not 'P_est' in _kwargs:
                _P = _kwargs['P']
                _kwargs['P_est'] = _P + np.random.randn(*_P.shape)*sd
            model = cls(**_kwargs)
            t, _a, u, u_est = model.simulate(g, T=T, verbose=False)
            if j == m:
                a = _a
            err.append(model.err[::t_sample])
            Pe.append(model.P_est)
        errm = err[m]
    else:
        model = cls(**kwargs)
        t, a, u = model.simulate(g, T=T, verbose=False)
        err, errm = None, None
        Pe = [P]
    ta = t.copy()
    t = t[::t_sample]
    return t, ta, a, err, errm, name, Pe

def plot(res, models, T, title, t_ex):
    fig, axs = plt.subplots(len(models), 3, sharex='col', sharey='col',
                            figsize=(10,7), width_ratios=[4,4,1])
    for t, ta, a, err, errm, name, Pe in res:
        t /= 24
        i = list(models.keys()).index(name)
        if err is not None:
            axs[i,1].fill_between(t, *np.quantile(err, [.1, .9], axis=0),
                                    color='k', lw=0, alpha=.2)
            axs[i,1].plot(t, errm, c='b', lw=1, ls='--',
                            label='MSE of example trajectory')
            axs[i,1].plot(t, np.median(err, axis=0), c='r', lw=1,
                            label=f'median MSE of {n} runs')
            axs[i,1].yaxis.tick_right()
            axs[i,1].set_ylabel(r'$\langle \Pi - \hat{Pi} \rangle$')
            axs[i,1].yaxis.set_label_position('right')

            Pe = np.median((np.array(Pe) - P), axis=0)
            axs[i,2].matshow(Pe, cmap='seismic', vmin=-1, vmax=1)
            c = lambda v : 'k' if abs(v) < .2 else 'w'
        else:
            Pe = Pe[0]
            axs[i,2].matshow(Pe, cmap='viridis')
            c = lambda v : 'w' if v < .5 else 'k'
            axs[i,2].set_title(r'real $\Pi$')
        for k in range(Pe.shape[0]):
            for l in range(Pe.shape[1]):
                e = Pe[l][k]
                axs[i,2].text(k, l, f'{e:.2f}', size='x-small',
                                  va='center', ha='center', color=c(e))

        mask = ta > T-t_ex
        ta /= 24
        axs[i,0].plot(ta[mask], np.clip(a[mask], a_min=0, a_max=None),
                      label=['HH', 'leisure', 'work'])
        axs[i,0].set_title(f'{name}:', loc='left')
        axs[i,0].set_ylabel(r'$\vec{a}^{>0}$')
    for ax in axs[-1,:2]:
        ax.set_xlabel('time [days]')
    for ax in axs[:,-1]:
        ax.set(xticks=[], yticks=[])
    axs[0,1].set_axis_off()
    axs[0,1].set_yscale('log', base=10)
    axs[1,2].set_title(r'MSE of $\hat{\Pi}$')
    fig.suptitle(title, fontsize='large')
    fig.tight_layout()
    handles, labels = axs[0,0].get_legend_handles_labels()
    handles2, labels2 = axs[1,1].get_legend_handles_labels()
    handles = handles + handles2
    labels = labels + labels2
    axs[0,1].legend(handles, labels, loc='upper left',
                    ncols=2, title='actions', alignment='left',
                    columnspacing=4, frameon=False)
    return fig, axs

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    t_ex = 100
    dt = 1e-1
    sd = 1
    T = 2000
    n = 5

    # models, name, title = get_base_model_comparison()
    models, name, title = get_delay_comparison(delays=[1, 5, 20], dt=dt)
    # models, name, title = get_noise_comparison()
    # models, name, title = compare_prob()

    # P, G, mu, g = get_feedback_model_pars(g12=5, mu3=20)
    # name += '-feedback-model'
    # name += '-test2'
    # P *= 1
    # models, name, title = get_wrong_P_comparison()
    # t_ex = T
    models, name, title = get_g_bias_comparison()

    pickle_fl = Path('data') / f'{name}.pickle'
    if not force and pickle_fl.exists():
        with pickle_fl.open('rb') as fl:
            res = pickle.load(fl)
    else:
        with mp.Pool(4) as p:
            res = p.map(run, models.items())
        with pickle_fl.open('wb') as fl:
            pickle.dump(res, fl)

    fig, axs = plot(res, models, T, title, t_ex)
    axs[0,0].set_ylim(-1, 7)
    fig.savefig(path / f'{name}.png')
    plt.show()

