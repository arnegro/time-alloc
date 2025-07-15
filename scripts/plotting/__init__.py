import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

pt_to_inch = lambda pt : pt / 72

textwidth = pt_to_inch(430)
colwidth = textwidth / 2

fullwidth = textwidth + 2
halfwidth = colwidth + 1
full_l = 1 / fullwidth
full_r =  1 - 1 / fullwidth
half_r = 1 - 1 / halfwidth
half_l = 1 / halfwidth

tex_dir = Path('..') / 'overleaf'
fig_dir = tex_dir / 'figures'
 
PGF = 'pgf' in sys.argv
if not PGF:
    img_file_format = 'png'
else:
    plt.rcParams['backend'] = 'pgf'
    img_file_format = 'pgf'

plt.rc('figure', dpi=300)

plt.rc('xtick', labelsize='xx-small')    # fontsize of the tick labels
plt.rc('ytick', labelsize='xx-small')    # fontsize of the tick labels
plt.rc('axes', labelsize='x-small')       # fontsize of the axis labels
plt.rc('axes', linewidth=.5)
plt.rc('font', size=10, family='serif')

def save_plot(name: str, directory: str = fig_dir, fig=None,
              _format=None, **kwargs):
    if _format is None:
        _format = img_file_format
    # if _format == 'pgf':
        # directory /= 'pgf'
    name = f'{name}.{_format}'
    directory.mkdir(exist_ok=True, parents=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(directory / name, **kwargs)
    return directory / name

def despine(ax, which=['top', 'right']):
    if which == 'all':
        which = ['top', 'bottom', 'left', 'right']
    if type(ax) in [list, np.ndarray]:
        for axi in ax:
            despine(axi, which=which)
    else:
        for spine in which:
            ax.spines[spine].set_visible(False)

##############################################################################

