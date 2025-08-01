import numpy as np
from matplotlib import pyplot as plt
from model import Model, PiLearnModelUdelayProb, PiLearnModelUdelay
from setup.models import get_base_pars

if __name__ == '__main__':
    P, G, mu, g = get_base_pars()

    dt = 1e-1
    P_est = P.copy() + np.random.randn(*P.shape)*0
    P_est[-1,0] /= 2
    basemodel = Model(P=P, G=G, mu=mu, dt=dt)
    U = np.array([[ 0.00015335, -0.00043222, -0.00016109],
                  [-0.00043222,  0.00258718, -0.00021393],
                  [-0.00016109, -0.00021393,  0.00051552]])
    U_inv = np.linalg.inv(U) * 1e0
    bayesmodel = PiLearnModelUdelayProb(P=P, G=G, mu=mu, P_est=P_est, dt=dt,
                                        delay=50, U_inv=U_inv)
    # bayesmodel = PiLearnModelUdelay(P=P, G=G, mu=mu, P_est=P_est, dt=dt,
                                        # delay=25, eta=1e-8)
    print(np.linalg.inv(bayesmodel.U_inv))
    print(np.linalg.trace(np.linalg.inv(bayesmodel.U_inv)))
    print(bayesmodel.P_est)

    T1 = 1000
    t1, a1, u1 = basemodel.simulate(g=g, T=T1)
    # t1, a1, u1 = np.array([1]), np.ones((1,3)), np.ones((1,3))
    t1, ab1, ub1, uest1 = bayesmodel.simulate(g=g, T=T1)
    print(np.linalg.inv(bayesmodel.U_inv))
    print(np.linalg.trace(np.linalg.inv(bayesmodel.U_inv)))
    err1 = bayesmodel.err
    print(bayesmodel.P_est)


    T2 = 2
    # P[-1,:] *= -1
    # basemodel.P = P.copy()
    # bayesmodel.P = P.copy()
    t2, a2, u2 = basemodel.simulate(g=g, T=T2, a0=a1[-1], u0=u1[-1])
    t2, ab2, ub2, uest2 = bayesmodel.simulate(g=g, T=T2, a0=ab1[-1],
                                              u0=ub1[-1], uest0=uest1[-1])
    t2 += t1.max()
    t = np.concatenate([t1, t2])
    a = np.concatenate([a1, a2], axis=0)
    ab = np.concatenate([ab1, ab2], axis=0)
    ub = np.concatenate([ub1, ub2], axis=0)
    uest = np.concatenate([uest1, uest2], axis=0)
    err = np.concatenate([err1, bayesmodel.err])

    fig, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(t, np.clip(a, a_min=0, a_max=None))
    axs[1].plot(t, np.clip(ab, a_min=0, a_max=None))
    # axs[1].plot(t, uest)#np.clip(ub, a_min=0, a_max=None))
    # axs[0].plot(t, ub)#np.clip(ub, a_min=0, a_max=None))
    axs[2].plot(t, err)

    w = np.cumsum(np.dot(np.clip(a, a_min=0, a_max=None), P[-1]))
    wb = np.cumsum(np.dot(np.clip(ab, a_min=0, a_max=None), P[-1]))
    axs[3].plot(t, w)
    axs[3].plot(t, wb)
    ker = np.ones(int(7*24/dt))
    ker /= ker.sum()
    wm = np.convolve(w, ker, mode='same')
    wbm = np.convolve(wb, ker, mode='same')
    axs[3].plot(t, wm)
    axs[3].plot(t, wbm)

    axs[2].set(yscale='log', ylabel='error', xlabel='time [h]')
    axs[1].set(ylabel='a$^{>0}$')
    axs[0].set(ylabel='u$')
    # axs[1].set(ylim=axs[0].get_ylim())
    
    fig.savefig(f'figures/P_shock_Uinv={U_inv.mean():.5f}.png')

    plt.show()
