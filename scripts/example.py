##############################################################################

""" Example script for usage """


##############################################################################
## IMPORTS ###################################################################

# external
import numpy as np
from matplotlib import pyplot as plt

# internal
from model import Model, PiLearnModelUdelay, PiLearnModelUdelayProb
from setup.models import get_base_pars


##############################################################################
## PARAMETERS ################################################################

pi_w = 0         # off diagonal Pi_23

gamma = 3        # gamma, off diagionals Gamma_12 / 4 = Gamma_21 / 4

dt = 5e-2        # simulation timestep in hours

T = 100          # simulation time in hours

sigma_u = None   # intensity of gaussian noise on u

delay = 5        # observation spacing in hours

g_bias = None    # no bias in belief over external influences (Sec. 3.5)
                 #  -> could be e.g. np.array([1, 0, 0]) for unknown
                 #     contribution to HH driving forces


##############################################################################
## DEFINITIONS ###############################################################

def simulate_model(model_cls, P, G, mu, g, sigma_u=None, T=500, dt=5e-2, 
                   **kwargs):
    model = model_cls(P, G, mu, sigma_u=sigma_u, dt=dt, **kwargs)
    res = model.simulate(g, T=T)
    a = np.clip(res[1], a_min=0, a_max=None)
    return model, res[0], a, *res[2:]

def simulate_base_model(*args, **kwargs):
    return simulate_model(Model, *args, **kwargs)

def simulate_bayes_model(P, P_est, G, mu, g, delay=5, g_bias=None,
                         init_certainty=1e-2, **kwargs):
    U_inv = np.eye(P.shape[0]) * init_certainty
    return simulate_model(PiLearnModelUdelayProb, P, G, mu, g, P_est=P_est,
                          delay=delay, g_bias=g_bias, U_inv=U_inv, **kwargs)

def simulate_gd_model(P, P_est, G, mu, g, delay=5, g_bias=None, eta=1e-3,
                      **kwargs):
    return simulate_model(PiLearnModelUdelay, P, G, mu, g, P_est=P_est,
                          delay=delay, g_bias=g_bias, eta=eta, **kwargs)

def plot(t, a, ab, agd, err_b, err_gd):
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10,8))

    for ax, _a, model in zip(axs[:-1],
                             [a, ab, agd],
                             ['base', 'Bayes', 'GD']):
        ax.plot(t, _a)
        ax.set_ylabel(r'$\vec{a}^{> 0}$ '+f'({model})')

    axs[3].plot(t, err_b, c='g', label='Bayes')
    axs[3].plot(t, err_gd, c='b', label='GD')
    axs[3].set(yscale='log', xlabel='time [h]', ylabel='MSE')
    axs[3].legend(frameon=False, loc='lower left')

    plt.show()

##############################################################################
## MAIN ######################################################################

def main():

    ## INIT ##################################################################

    # model parameters as described by Eq. (10) in report of basic
    # HH--leisure--work model:
    # \Pi, \Gamma, \mu, g(t)
    P, G, mu, g = get_base_pars(gamma=gamma, pi=pi_w)

    # initial \hat{\Pi}
    P_est = P + np.random.randn(*P.shape)

    ## STANDARD MODEL ########################################################

    # simulate model without learning
    _, t, a, u = simulate_base_model(P, G, mu, g, sigma_u=sigma_u, T=T, dt=dt)


    ## LEARNING MODELS #######################################################

    # simulate bayesian learner
    bayes_model, t, ab, ub, u_estb = simulate_bayes_model(P, P_est, G, mu, g,
                                                delay=delay, sigma_u=sigma_u,
                                                g_bias=g_bias, T=T, dt=dt,
                                                init_certainty=1e-2)
    err_b = bayes_model.err

    # simulate gradient descent learner
    gd_model, t, agd, ugd, u_estgd = simulate_gd_model(P, P_est, G, mu, g,
                                                delay=delay, sigma_u=sigma_u,
                                                g_bias=g_bias, T=T, dt=dt,
                                                eta=1e-2)
    err_gd = gd_model.err


    ## PLOT ##################################################################

    plot(t, a, ab, agd, err_b, err_gd)

##############################################################################

if __name__ == '__main__':
    main()

##############################################################################
