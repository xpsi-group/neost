# Standard libraries
import pathlib
import warnings

# 3rd party
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns

# TeX fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams['font.family'] = 'serif'

def contour_levels(grid, levels=[0.68, 0.95]):
    """Compute contour levels for a gridded 2D posterior"""
    sorted_ = np.flipud(np.sort(grid.ravel()))
    pct = np.cumsum(sorted_) / np.sum(sorted_)
    cutoffs = np.searchsorted(pct, np.array(levels))
    return np.sort(sorted_[cutoffs])

def plot_joint_distribution(ax, data, colors, fill, ls=None, bins=10, levels=[0.68, 0.95], lw=0.8, limits=None):
    x1 = data[:,1]
    x2 = data[:,0]
    hist, xedges, yedges = np.histogram2d(x1, x2, bins=bins)
    hist = hist.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    levels = contour_levels(hist, levels)
    if fill:
        ax.contourf(hist, extent=extent, colors=colors, levels=levels, extend='max')
    else:
        ax.contour(hist, extent=extent, colors=colors, linewidths=lw, levels=levels, linestyles=ls)

def plot_scatter(ax, data, colors, s=0.1, alpha=0.1, label=None):
    ax.scatter(data[:,1], data[:,0], color=colors[1], s=s, alpha=alpha, label=label)

def plot_KDE(ax, data, colors, fill, ls='solid', bw_adjust=1.5):
    levels = [0.05, 0.32, 1]
    lw = 0.8 if ls != 'dotted' else 1.0
    lw = None if fill else lw
    gridsize = 40
    alpha = 1.0
    if fill:
        # Had to split these up because cmap behaves differently for filled and non-filled contours.
        # Using 'cmap=ListedColors(colors)' with fill=False colors all lines the same
        # Tested with seaborn 0.11.1 and 0.13.2, matplotlib 3.8.4
        sns.kdeplot(x=data[:,1], y=data[:,0], gridsize=gridsize, bw_adjust=bw_adjust, fill=fill, ax=ax, levels=levels, alpha=alpha, cmap=ListedColormap(colors), label=None)
    else:
        sns.kdeplot(x=data[:,1], y=data[:,0], gridsize=gridsize, bw_adjust=bw_adjust, fill=fill, ax=ax, levels=levels, alpha=alpha, colors=colors, label=None, linewidths=lw, linestyles=ls)

def get_quantiles_log10(pressures, quantiles=[0.025, 0.16, 0.84, 0.975], warn_small=False):
    energydensities = np.log10(np.logspace(14.2,16,50))
    x = []
    contours = []
    for i, row in enumerate(pressures):
        pruned = row[np.where(row > 0)]
        pruned = np.log10(pruned)
        if len(pruned) < 10:
            if warn_small:
                print(f'Sample too small (length: {len(pruned)}), ignoring')
        else:
            x.append(energydensities[i])
            contours.append(np.quantile(pruned, quantiles))
    return x, np.array(contours)

def plot_1D_quantiles(ax, y, colors, ls):
    x, contours = get_quantiles_log10(y)
    assert(contours.shape[1] == 4)
    if ls == 'filled':
        ax.fill_between(x, contours[:,0], contours[:,3], color=colors[0])
        ax.fill_between(x, contours[:,1], contours[:,2], color=colors[1])
    else:
        lw = 0.8 if ls != 'dotted' else 1.0
        ax.plot(x, contours[:,0], color=colors[0], ls=ls, lw=lw)
        ax.plot(x, contours[:,1], color=colors[1], ls=ls, lw=lw)
        ax.plot(x, contours[:,2], color=colors[1], ls=ls, lw=lw)
        ax.plot(x, contours[:,3], color=colors[0], ls=ls, lw=lw)

def custom_line(color, linestyle=None, lw=None):
    if lw is None:
        return Line2D([0], [0], color=color, linestyle=linestyle)
    else:
        return Line2D([0], [0], color=color, linestyle=linestyle, lw=lw)

def custom_patch(color, linestyle=None, alpha=1.0, lw=None):
    return mpatches.Patch(color=color, alpha=alpha, lw=4)

def double_interval_legend(color1, color2, lw=4):
    return (custom_patch(color1, lw=lw), custom_line(color2, lw=lw))

def right_side_ticks(ax):
    '''Add in-pointing ticks on the right side'''
    ax2 = ax.secondary_yaxis('right')
    ax2.tick_params(axis='y', which='both', direction='in')
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticks(ax.get_yticks(minor=True), minor=True)
    ax2.set_yticklabels([])
    ax2.set_zorder(20)

def top_side_ticks(ax):
    '''Add in-pointing ticks on the top side'''
    ax2 = ax.secondary_xaxis('top')
    ax2.tick_params(axis='x', which='both', direction='in')
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticks(ax.get_xticks(minor=True), minor=True)
    ax2.set_xticklabels([])
    ax2.set_zorder(20)

def save(fig, path, fname, tmp=False):
    fname = 'tmp.pdf' if tmp else fname
    path = '.' if tmp else path
    if not tmp:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    print(f'Saving to {path}/{fname}')
    fig.savefig(f'{path}/{fname}', bbox_inches='tight')


# Color definitions
c_n3lo = ['xkcd:baby blue', 'xkcd:electric blue']
c_n2lo = ['xkcd:orangish', 'xkcd:tomato']
c_hebeler = ['xkcd:black', 'xkcd:black']
c_n3lo_11 = ['xkcd:azure', 'xkcd:cobalt blue']

tmp_color = sns.cubehelix_palette(8, start=.5, rot=-.75, dark=0.2, light=.85)[0::3]
c_n3lo_posterior = tmp_color[:2]
