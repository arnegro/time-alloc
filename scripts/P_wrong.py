import numpy as np
from matplotlib import pyplot as plt
from model import Model, PiLearnModelUdelayProb, PiLearnModelUdelay
from setup.models import get_base_pars

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    dt = 1e-1
    P_est = P + np.random.randn(*P.shape)*0
    P_est[-1,:] *= -1
    basemodel = Model(P=P, G=G, mu=mu, dt=dt)
    bayesmodel = PiLearnModelUdelayProb(P=P, G=G, mu=mu, P_est=P_est, dt=dt,
                                        delay=5)
    gradmodel1 = PiLearnModelUdelay(P=P, G=G, mu=mu, P_est=P_est, dt=dt,
                                    delay=25, eta=1e-10)
    gradmodel2 = PiLearnModelUdelay(P=P, G=G, mu=mu, P_est=P_est, dt=dt,
                                    delay=25, eta=1e-3)
    # print(np.linalg.inv(bayesmodel.U_inv))

    T = 3000
    fig, axs = plt.subplots(5, 1, sharex=True)
    t, a, u = basemodel.simulate(g=g, T=T)
    axs[0].plot(t, np.clip(a, a_min=0, a_max=None))
    for i, model in enumerate([bayesmodel, gradmodel1, gradmodel2]):
        t, a, u, uest = model.simulate(g=g, T=T)
        axs[i+1].plot(t, np.clip(a, a_min=0, a_max=None))
        axs[-1].plot(t, model.err)
        axs[i+1].set(ylim=axs[0].get_ylim())

    axs[-1].set(yscale='log')

    plt.show()
