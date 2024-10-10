# Stardard libraries
import argparse
import os
import pathlib
import sys

# 3rd party
import matplotlib.pyplot as plt
import numpy as np

# Local
import plotting

# Plotting function
plot = plotting.plot_1D_quantiles

# Paths
current_path = os.path.dirname(__file__)
parent = os.path.dirname(current_path)

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tmp_output', action='store_true')
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

# Create subplots, remove spaces
width = 7
height = 5
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(width,height))
fig.subplots_adjust(wspace=0, hspace=0)

# Set axis limits and ticks
xticks1 = [14.4, 14.6, 14.8, 15.0]
xticks2 = [14.4, 14.6, 14.8, 15.0, 15.2]
[ax.set_xticks(xticks1) for ax in axes[:,0]]
[ax.set_xticks(xticks2) for ax in axes[:,1]]
[ax.set_xticklabels([]) for ax in axes[:3,:].flatten()]
axes[-1,0].set_xticklabels(xticks1)
axes[-1,1].set_xticklabels(xticks2)

yticks = [33.5, 34.0, 34.5, 35.0, 35.5]
[ax.set_yticks(yticks) for ax in axes.flatten()]
[ax.set_yticklabels(yticks) for ax in axes[:,0]]
[ax.set_yticklabels([]) for ax in axes[:,1]]

[ax.set_xlim(14.4,15.2) for ax in axes.flatten()]
[ax.set_ylim(33.4,35.8) for ax in axes.flatten()]

# Set labels and titles
xlabel = r'$\log_{10}(\varepsilon)$ [g/cm$^3$]'
ylabel = r'$\log_{10}(P)$ [dyn/cm$^2$]'
axes[-1,0].set_xlabel(xlabel)
axes[-1,1].set_xlabel(xlabel)
[ax.set_ylabel(ylabel) for ax in axes[:,0]]
axes[0,0].set_title('PP')
axes[0,1].set_title('CS')

# Define colors
c_posterior = plotting.c_n3lo_posterior
c_n2lo = plotting.c_n2lo

# Results directory
directory = f'{parent}/results' if not args.repro else f'{parent}/repro'

# Prior files
pp_n3lo_15_prior = np.load(f'{directory}/prior/pp/n3lo/15/pressures.npy')
cs_n3lo_15_prior = np.load(f'{directory}/prior/cs/n3lo/15/pressures.npy')

# Posterior files
pp_n3lo_15_baseline = np.load(f'{directory}/posterior/pp/n3lo/15/baseline/pressures.npy')
pp_n3lo_15_new = np.load(f'{directory}/posterior/pp/n3lo/15/new/pressures.npy')

cs_n3lo_15_baseline = np.load(f'{directory}/posterior/cs/n3lo/15/baseline/pressures.npy')
cs_n3lo_15_new = np.load(f'{directory}/posterior/cs/n3lo/15/new/pressures.npy')

# PP 1.5
ax = axes[0,0]
plot(ax, pp_n3lo_15_baseline, c_posterior, 'filled')
plot(ax, pp_n3lo_15_prior, c_n2lo, 'dotted')
ax = axes[1,0]
plot(ax, pp_n3lo_15_new, c_posterior, 'filled')
plot(ax, pp_n3lo_15_prior, c_n2lo, 'dotted')

# CS 1.5
ax = axes[0,1]
plot(ax, cs_n3lo_15_baseline, c_posterior, 'filled')
plot(ax, cs_n3lo_15_prior, c_n2lo, 'dotted')
ax = axes[1,1]
plot(ax, cs_n3lo_15_new, c_posterior, 'filled')
plot(ax, cs_n3lo_15_prior, c_n2lo, 'dotted')

# Add x*n_0
n0 = 2.68e14 # n0 in g/cm^3
[ax.axvline(np.log10(1.5*n0), color='xkcd:gray', lw=0.5) for ax in axes.flatten()]

# Add text
x = 0.03
y = 0.88
txt = r'N$^3$LO $\chi$EFT $\leq 1.5 n_0$'
[ax.text(x, y, txt, transform=ax.transAxes) for ax in axes[:,0]]

ax = axes[0,0]
txt = "Baseline"
ax.text(x, y-0.1, txt, transform=ax.transAxes)

ax = axes[1,0]
txt = "New"
ax.text(x, y-0.1, txt, transform=ax.transAxes)

# Minor ticks
[ax.minorticks_on() for ax in axes.flatten()]

# Ticks on right and top side
[plotting.right_side_ticks(ax) for ax in axes[:,1]]
[plotting.top_side_ticks(ax) for ax in axes[0,:]]

# Z-order the panels. Needed for the ticks to show up properly
# Right on top of left, top on top of bottom.
axes[1,0].set_zorder(5)
axes[1,1].set_zorder(6)
axes[0,0].set_zorder(7)
axes[0,1].set_zorder(8)

# Add legends
loc = 'upper left'
line1 = plotting.custom_line(c_n2lo[1], 'dotted', lw=1)
line2 = plotting.double_interval_legend(*c_posterior)
custom_lines = [line1, line2]
[ax.legend(custom_lines, ['Prior', 'Posterior'], loc=loc, frameon=False) for ax in axes[:,1]]

# Save result
plotting.save(fig, f'{parent}/fig', 'P_eps_posteriors.pdf', args.tmp_output)
