from matplotlib import pyplot as plt
import numpy as np
from model import Model
from setup.models import get_feedback_model_pars
from model.learning_models import PiLearnModelUdelay, PiLearnModelUdelayProb

def dynamics_plot():
    gs = np.linspace(0, 10, 5)
    fig, axs = plt.subplots(len(gs), 2, sharex=True, sharey=True,
                            figsize=(10,6))
    for i, g12 in enumerate(gs):
        P, G, mu, g = get_feedback_model_pars(g12=g12, mu3=0)
        model = Model(P, G, mu, dt=1e-1)
        t, a, u = model.simulate(g, T=200)

        # axs[i,0].plot(t, np.clip(u, a_min=mu, a_max=None))
        for j in range(a.shape[1]):
            axs[i,0].plot(t[t>100], np.clip(a[t>100,j], a_min=0, a_max=None),
                        label=rf'$i={j}$')
        axs[i,0].text(.99, .99, r'$\gamma_{12} ='+f'{g12:.2f}$',
                      transform=axs[i,0].transAxes, ha='right', va='top')
    axs[0,0].set(xlim=(100, 200),
                 title=r'$\mu_3=0, \:\: \gamma_{23}=\gamma_{31}=10$')
    axs[0,0].legend(ncols=3, frameon=False)
    for ax in axs[:,0]:
        ax.set_ylabel(r'$\vec{a}^{>0}$')

    mus = np.linspace(15, 25, len(gs))
    for i, mu3 in enumerate(mus):
        P, G, mu, g = get_feedback_model_pars(g12=5, mu3=mu3)
        model = Model(P, G, mu, dt=1e-1)
        t, a, u = model.simulate(g, T=200)

        # axs[i,0].plot(t, np.clip(u, a_min=mu, a_max=None))
        axs[i,1].plot(t[t>100], np.clip(a[t>100], a_min=0, a_max=None))
        axs[i,1].text(.99, .99, rf'$\mu_3 = {mu3:.2f}$',
                      transform=axs[i,1].transAxes, ha='right', va='top')
    axs[0,1].set(xlim=(100, 200),
            title=r'$\gamma_{12}=5, \:\: \gamma_{23}=\gamma_{31}=10$')
    axs[-1,0].set_xlabel('time [h]')
    axs[-1,1].set_xlabel('time [h]')
    fig.tight_layout()
    fig.savefig('figures/feedback_model-dynamics_plot.png')
    plt.show()
    


if __name__ == '__main__':

    dynamics_plot()
    quit()

    P, G, mu, g = get_feedback_model_pars(g12=5, mu3=20)
    P_est = P + np.random.randn(*P.shape)*1
    model = PiLearnModelUdelay(P, G, mu, P_est=P_est,
                               dt=1e-1, delay=1, eta=1e-3)
    # model = PiLearnModelUdelayProb(P, G, mu, P_est=P_est,
                                   # dt=1e-1, delay=20)
    t, a, u, u_est = model.simulate(g, T=1000)
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t[t>800], np.clip(u[t>800], a_min=mu, a_max=None))
    axs[1].plot(t[t>800], np.clip(a[t>800], a_min=0, a_max=None))
    axs[2].plot(t, model.err)
    axs[2].set(yscale='log')
    plt.show()
