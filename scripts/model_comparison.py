from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
from model import Model
from setup.models import get_base_pars
from setup.model_comparison \
        import get_base_model_comparison, get_delay_comparison, \
               get_noise_comparison

path = Path('figures')

def run(arg):
    name, (cls, kwargs) = arg
    if cls != Model:
        m = np.random.randint(n)
        err, Pe = [], []
        for j in range(n):
            print(f'[{j+1}/{n}] {name}', end='    \r')
            P_est = P + np.random.randn(*P.shape)*sd
            model = cls(P, G, mu, P_est=P_est, dt=dt, **kwargs)
            t, _a, u, u_est = model.simulate(g, T=T, verbose=False)
            if j == m:
                a = _a
            err.append(model.err)
            Pe.append(model.P_est)
        errm = err[m]
    else:
        model = cls(P, G, mu, dt=dt, **kwargs)
        t, a, u = model.simulate(g, T=T, verbose=False)
        err, errm = None, None
        Pe = [P]
    return t, a, err, errm, name, Pe

def plot(res, models, T, title):
    fig, axs = plt.subplots(len(models), 3, sharex='col', sharey='col',
                            figsize=(10,7), width_ratios=[4,4,1])
    for t, a, err, errm, name, Pe in res:
        i = list(models.keys()).index(name)
        if err is not None:
            axs[i,1].fill_between(t, *np.quantile(err, [.1, .9], axis=0),
                                    color='k', lw=0, alpha=.2)
            axs[i,1].plot(t, errm, c='b', lw=1, ls='--',
                            label='MSE of example trajectory')
            axs[i,1].plot(t, np.median(err, axis=0), c='r', lw=1,
                            label=f'median MSE of {n} runs')
            axs[i,1].yaxis.tick_right()
            axs[i,1].set_ylabel(r'$\langle (\Pi - Pi_e)^2 \rangle$')
            axs[i,1].yaxis.set_label_position('right')

            Pe = np.median((np.array(Pe) - P)**2, axis=0)
            axs[i,2].matshow(Pe, cmap='Reds', vmin=0, vmax=1)
            c = lambda v : 'k' if v < .5 else 'w'
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

        mask = t > T-200
        axs[i,0].plot(t[mask], np.clip(a[mask], a_min=0, a_max=None),
                        label=['HH', 'leisure', 'work'])
        axs[i,0].set_title(f'{name}:', loc='left')
        axs[i,0].set_ylabel(r'$\vec{a}^{>0}$')
    for ax in axs[-1,:2]:
        ax.set_xlabel('time')
    for ax in axs[:,-1]:
        ax.set(xticks=[], yticks=[])
    axs[0,1].set_axis_off()
    axs[0,1].set_yscale('log')
    axs[1,2].set_title(r'MSE of $\Pi_e$')
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

    models, name, title = get_base_model_comparison()
    models, name, title = get_delay_comparison()
    # models, name, title = get_noise_comparison()

    sd = 1
    T = 5000
    n = 100
    dt = 1e-1

    with mp.Pool(4) as p:
        res = p.map(run, models.items())

    fig, axs = plot(res, models, T, title)
    fig.savefig(path / f'{name}.png')
    plt.show()

