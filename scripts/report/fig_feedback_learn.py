import numpy as np
from matplotlib import pyplot as plt
from model import PiLearnModelUdelay, PiLearnModelUdelayProb, \
                  GFeedbackExtension, Model
from setup.models import get_base_pars
from plotting import halfwidth, save_plot, half_l

def smooth(y, k):
    y = np.concatenate([[y[0]]*3*k, y, [y[-1]]*3*k])
    ker = np.ones(k) / k
    ys = np.convolve(y, ker, mode='same')
    return ys[3*k:-3*k]

def fig_weekend_stim():
    P, G, mu, g = get_base_pars(gamma=1)
    G[1,2] = 4
    G[2,1] = 4
    g_bar = np.array([2., 0, 0])
    mu = np.zeros_like(g_bar)
    tau = 50
    dt = 1e-1
    T1, T2 = 2000, 0
    g1 = lambda t : np.array([0, 3*(int(t/24) % 7 == 0), 0])
    g2 = lambda t : np.array([0, 0, 0])
    fig, axs = plt.subplots(2, 2, sharex='col', width_ratios=[2, 1],
                            figsize=(halfwidth, 2))
    betas = np.linspace(0, 3, 10)
    cmap = plt.get_cmap()
    get_color = lambda b : cmap((b - betas.min()) / (betas.max() - betas.min()))
    k = int(24/dt)
    errs, P_ests = [], []
    for beta in betas:
        P_est = P + np.random.randn(*P.shape)*0
        model = GFeedbackExtension.make(PiLearnModelUdelayProb, P=P, G=G,
                                        P_est=P_est, mu=mu, g_bar=g_bar,
                                        tau=tau, beta=beta, dt=dt)
        t, a, u, u_est, g_fb = model.simulate(g1, T=T1)
        if beta in betas[::3]:
            a = a[:,1]
            g_fb = g_fb[:,1]
            a = np.clip(a, a_min=0, a_max=None)
            axs[0,0].plot(t/24/30, smooth(a, k), c=get_color(beta), lw=.5)
            axs[1,0].plot(t/24/30, smooth(g_fb, k), c=get_color(beta), lw=.5)
        errs.append(((u - u_est)[-int(100/dt)]**2).mean())
        P_ests.append(model.P_est[1] - model.P[1])
    for i, ax in enumerate(axs[:,0]):
        y0, y1 = ax.get_ylim()
        ax.fill_between(t/24/30, y0-1, 
                        [(y1+2-y0)*(g1(_t)[1]>0) + y0-1 for _t in t],
                        alpha=.1, lw=0, color='k')
        ax.set_ylim(y0, y1)
        ax.text(.02, .98, f'(a{"i"*(i+1)})', size='xx-small', ha='left',
                va='top', transform=ax.transAxes)
    axs[0,1].text(.9, .02, '(bi)', size='xx-small', ha='right', va='bottom',
                  transform=axs[0,1].transAxes)
    axs[1,1].text(.98, .02, '(bii)', size='xx-small', ha='right', va='bottom',
                  transform=axs[1,1].transAxes)
    axs[0,1].plot(betas, P_ests, lw=.5,
                  label=[r'$\Delta \hat{\Pi}_{2'+str(i+1)+'}$' for i in range(3)])
    axs[0,1].legend(frameon=False, fontsize=5, loc='lower left')
    axs[1,1].plot(betas, errs, c='k', lw=.5)
    axs[0,0].set(ylabel='$a_2$')
    axs[1,0].set(ylabel='intrinsic $g_2$', xlabel='time [months]')
    axs[1,1].set(xlabel=r'$\beta$', yscale='log',
                 xlim=(betas.min(), betas.max()))
    axs[1,1].text(.02, .98, r'MSE$(\hat{u}_2)$', size='xx-small',
                  ha='left', va='top', transform=axs[1,1].transAxes)
    cax = axs[0,1].inset_axes([0,1,1,.03])
    im = cax.imshow([[]], vmin=min(betas), vmax=max(betas), cmap=cmap,
                    aspect='auto')
    plt.colorbar(im, cax=cax, orientation='horizontal')
    cax.set(xticks=[])
    fig.subplots_adjust(top=0.974, bottom=0.173, left=half_l, right=0.99,
                        hspace=0.301, wspace=0.31)
    save_plot('fig_weekend_stim')
    plt.show()


if __name__ == '__main__':
    fig_weekend_stim()
    quit()
    P, G, mu, g = get_base_pars(gamma=0)
    G[1,2] = 0
    g = lambda t : np.array([0, 1*(t<2000), 0])
    g2 = lambda t : np.array([0, 0, 0])
    g_bar = np.array([2., 0, 0])
    beta = np.array([0, 3, 0])
    mu = np.zeros_like(beta)
    tau = 50
    dt = 1e-1
    P_est = P + np.random.randn(*P.shape)*1

    model = GFeedbackExtension.make(PiLearnModelUdelayProb, P=P, P_est=P_est,
                                    G=G, mu=mu, delay=50,# sigma_u=.1,
                                    g_bar=g_bar, tau=tau, beta=beta, dt=dt)
    # model = GFeedbackExtension.make(Model, P=P, G=G, mu=mu,
                                    # g_bar=g_bar, tau=tau, beta=beta, dt=dt)
    t, a, u, u_est, g_fb = model.simulate(g, T=2000)
    # t, a, u, g_fb = model.simulate(g, T=4000)
    print(model.P)
    print(model.P_est)
    print(np.linalg.trace(model.U_inv))

    fig, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(t, np.clip(a[:,1], a_min=0, a_max=None))
    axs[1].plot(t, [g(_t)[1] for _t in t])
    axs[1].plot(t, g_fb[:,1])
    axs[2].plot(t, u_est)
    axs[2].plot(t, u[:,1])
    axs[3].plot(t, (np.clip(u, a_min=mu, a_max=None)[:,1] - (G @ np.clip(a.T, a_min=0, a_max=None))[1]))
    t0 = t.max()
    model.U_inv *= 1e15
    # model.P_est = P
    t, a, u, u_est, g_fb = model.simulate(g2, T=5000, g0=model._g.copy(),
                                          u0=model.u.copy())
    print(model.P_est)
    t += t0
    axs[0].plot(t, np.clip(a[:,1], a_min=None, a_max=None))
    axs[1].plot(t, [g(_t)[1] for _t in t])
    axs[1].plot(t, g_fb[:,1])
    axs[2].plot(t, u_est)
    axs[2].plot(t, u[:,1])
    axs[3].plot(t, (np.clip(u, a_min=mu, a_max=None)[:,1] - (G @ np.clip(a.T, a_min=0, a_max=None))[1]))
    # axs[3].plot(t, np.clip(u, a_min=mu, a_max=None)[:,1])
    axs[3].plot(t, - (G @ np.clip(a.T, a_min=0, a_max=None))[1])
    plt.show()
