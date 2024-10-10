# Standard libraries
import argparse
import os
import pathlib

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Local
import plotting

# Paths
current_path = os.path.dirname(__file__)
parent = os.path.dirname(current_path)

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tmp_output', action='store_true')
parser.add_argument('-s', '--scatter', action='store_true')
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

style = 'scatter' if args.scatter else 'hexbin'

# Create subplots, remove spaces
width = 7.0
height = 3
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(width,height))
axes = axes.reshape((1,2))
fig.subplots_adjust(wspace=0, hspace=0)

# Set labels and titles
xlabel = '$R$ [km]'
ylabel = r'$M$ [$M_{\odot}$]'

[ax.set_xlabel(xlabel) for ax in axes.flatten()]
axes[0,0].set_ylabel(ylabel)
axes[0,0].set_title('PP')
axes[0,1].set_title('CS')

# Define colors
scatter_color = plotting.c_n3lo_posterior[1]
hexbin_cmap = 'OrRd_r'
hist_color = scatter_color if args.scatter else 'xkcd:tomato'

# Results directory
directory = f'{parent}/results/posterior' if not args.repro else f'{parent}/repro/posterior'

# Load data
data_pp = np.loadtxt(f'{directory}/pp/n3lo/15/new/MR_prpr.txt')
data_cs = np.loadtxt(f'{directory}/cs/n3lo/15/new/MR_prpr.txt')

R_14_pp = np.loadtxt(f'{directory}/pp/n3lo/15/new/table_data.txt')[:,5]
R_14_cs = np.loadtxt(f'{directory}/cs/n3lo/15/new/table_data.txt')[:,5]

if style == 'hexbin':
    gridsize = 60
    # PP
    ax = axes[0,0]
    ax.hexbin(data_pp[:,1], data_pp[:,0], gridsize=gridsize, cmap=hexbin_cmap, mincnt=1)

    # CS
    ax = axes[0,1]
    ax.hexbin(data_cs[:,1], data_cs[:,0], gridsize=gridsize, cmap=hexbin_cmap, mincnt=1)

elif style == 'scatter':
    alpha = 0.5
    s = 0.1
    # PP
    ax = axes[0,0]
    ax.scatter(data_pp[:,1], data_pp[:,0], marker='H', s=s, color=scatter_color, alpha=alpha)

    # CS
    ax = axes[0,1]
    ax.scatter(data_cs[:,1], data_cs[:,0], marker='H', s=s, color=scatter_color, alpha=alpha)


# Plot insets
location = [0.04, 0.13, 0.25, 0.28] # location of inset plots [left, bottom, width, height]
inset_fontsize = 5
labelpad = 1.5
histtype = 'stepfilled'

left, bottom, width, height = location
ax2 = axes[0,0].inset_axes([left, bottom, width, height])
ax2.hist(R_14_pp, bins=50, color=hist_color, histtype=histtype)
ax2.set_xlabel(r'$R_{1.4}$ [km]', fontsize=inset_fontsize, labelpad=labelpad)
ax2.set_yticks([])
ax2.set_xlim(11,13)
ax2.tick_params(labelsize=inset_fontsize)
ax2.minorticks_on()

left, bottom, width, height = location
ax2 = axes[0,1].inset_axes([left, bottom, width, height])
ax2.hist(R_14_cs, bins=50, color=hist_color, histtype=histtype)
ax2.set_xlabel(r'$R_{1.4}$ [km]', fontsize=inset_fontsize, labelpad=labelpad)
ax2.set_yticks([])
ax2.set_xlim(11,13)
ax2.tick_params(labelsize=inset_fontsize)
ax2.minorticks_on()

# Add line to indicate inset slice
x = np.linspace(11,13)
y = 1.4 * np.ones_like(x)
[ax.plot(x, y, color='xkcd:black', ls='dashed') for ax in axes.flatten()]

# Add text
ax = axes[0,0]
txt = r'N$^3$LO $\chi$EFT $\leq 1.5 n_0$'
ax.text(0.03, 0.90, txt, transform=ax.transAxes)
txt = 'New'
ax.text(0.03, 0.80, txt, transform=ax.transAxes)

# Set axis limits and ticks
r_min = 9
r_max = 14
m_min = 1.0
m_max = 2.6
xticks = [9, 10, 11, 12, 13, 14]
yticks = [m_min, 1.4, 1.8, 2.2]
[ax.set_xlim(r_min,r_max) for ax in axes.flatten()]
[ax.set_ylim(m_min,m_max) for ax in axes.flatten()]

axes[0,0].set_xticks(xticks[:-1])
axes[0,1].set_xticks(xticks)

[ax.set_yticks(yticks) for ax in axes.flatten()]
axes[0,0].set_yticks(yticks + [2.6])
axes[0,1].set_yticklabels([])

# Minor ticks
[ax.minorticks_on() for ax in axes.flatten()]

# Ticks on right and top side
[plotting.right_side_ticks(ax) for ax in axes[:,1]]
[plotting.top_side_ticks(ax) for ax in axes[0,:]]


# Z-order the panels. Needed for the ticks to show up properly
# Right on top of left, top on top of bottom.
axes[0,0].set_zorder(5)
axes[0,1].set_zorder(6)

# Save result
plotting.save(fig, f'{parent}/fig', 'MR_posterior_scatter.pdf', args.tmp_output)
