import numpy as np
from matplotlib import pyplot as plt
from plotting import halfwidth, save_plot, half_l

def beta_plot():
    n = 100
    fig, ax = plt.subplots(figsize=(halfwidth, 1))
    betas = np.linspace(-.5, 1, n)
    cmap = plt.get_cmap('plasma')
    ## set x of divergence to one: * pi at x
    ax.plot(betas, 1 / (1 - betas), label=r'$g$', c='k', ls='-')
    ## set x of divergence to one: * pi^2/4g at x
    ax.plot(betas, 2 / (1 + np.sqrt(1 - betas)), label=r'$\pi$',
            c='k', ls='--')
    ax.set(yscale='log', xlabel=r'$\beta / \beta_{c}$',
            ylabel=r'$a^* / a^*_0$')#,
            # title=r'rescaling of action fixed point under different $\beta$')
    ax.grid(lw=.2)
    legend = ax.legend(title='feedback via:', title_fontsize=8, fontsize=6,
                       ncols=2, frameon=True)
    ax.set_ylim(ax.get_ylim()[0], 100)

    legend.get_frame().set_facecolor('white')  # White background
    legend.get_frame().set_edgecolor('none')   # No border
    legend.get_frame().set_linewidth(0)        # No edge line
    legend.get_frame().set_alpha(1.0)          # Fully opaque
    # legend.set_shadow(False)                   # No shadow
    fig.subplots_adjust(top=0.955, bottom=0.344, left=half_l, right=0.99,
                        hspace=0.2, wspace=0.2)
    save_plot('fig_feedback_1d')
    plt.show()

if __name__ == '__main__':
    beta_plot()
