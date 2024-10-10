# Standard libraries
import argparse
import os
import pathlib
import sys

# 3rd party
import matplotlib.pyplot as plt
import numpy as np

# Local
import plotting

# Paths
current_path = os.path.dirname(__file__)
parent = os.path.dirname(current_path)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--numpy-histogram', action='store_true')
parser.add_argument('-s', '--scatter', action='store_true')
parser.add_argument('-t', '--tmp_output', action='store_true')
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

# Choose plotting function (KDE, numpy-histogram, or scatterplot)
plot = plotting.plot_KDE
if args.numpy_histogram:
    plot = plotting.plot_joint_distribution
elif args.scatter:
    plot = plotting.plot_scatter

# Create subplots, remove spaces
width = 7
height = 5
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(width,height))
fig.subplots_adjust(wspace=0, hspace=0)

# Set labels and titles
xlabel = '$R$ [km]'
ylabel = r'$M$ [$M_{\odot}$]'
axes[1,0].set_xlabel(xlabel)
axes[1,1].set_xlabel(xlabel)
axes[0,0].set_ylabel(ylabel)
axes[1,0].set_ylabel(ylabel)
axes[0,0].set_title('PP')
axes[0,1].set_title('CS')

# Define colors
c_n3lo = plotting.c_n3lo_posterior
c_n2lo = plotting.c_n2lo
c_hebeler = plotting.c_hebeler
c_n3lo_11 = plotting.c_n3lo_11

# Results directory
directory = f'{parent}/results/posterior' if not args.repro else f'{parent}/repro/posterior'

# PP
pp_n3lo_11_baseline = np.loadtxt(f'{directory}/pp/n3lo/11/baseline/MR_prpr.txt')
pp_n3lo_15_baseline = np.loadtxt(f'{directory}/pp/n3lo/15/baseline/MR_prpr.txt')
pp_n3lo_11_new = np.loadtxt(f'{directory}/pp/n3lo/11/new/MR_prpr.txt')
pp_n3lo_15_new = np.loadtxt(f'{directory}/pp/n3lo/15/new/MR_prpr.txt')

# CS
cs_n3lo_11_baseline = np.loadtxt(f'{directory}/cs/n3lo/11/baseline/MR_prpr.txt')
cs_n3lo_15_baseline = np.loadtxt(f'{directory}/cs/n3lo/15/baseline/MR_prpr.txt')
cs_n3lo_11_new = np.loadtxt(f'{directory}/cs/n3lo/11/new/MR_prpr.txt')
cs_n3lo_15_new = np.loadtxt(f'{directory}/cs/n3lo/15/new/MR_prpr.txt')

# PP
ax = axes[0,0]
plot(ax, pp_n3lo_15_baseline, c_n3lo, True)
plot(ax, pp_n3lo_11_baseline, c_n2lo, False, '-')

ax = axes[1,0]
plot(ax, pp_n3lo_15_new, c_n3lo, True)
plot(ax, pp_n3lo_11_new, c_n2lo, False, '-')

# CS
ax = axes[0,1]
plot(ax, cs_n3lo_15_baseline, c_n3lo, True)
plot(ax, cs_n3lo_11_baseline, c_n2lo, False, '-')

ax = axes[1,1]
plot(ax, cs_n3lo_15_new, c_n3lo, True)
plot(ax, cs_n3lo_11_new, c_n2lo, False, '-')

# Add text
txt = "Baseline"
ax = axes[0,0]
ax.text(0.03, 0.90, txt, transform=ax.transAxes)
txt = "New"
ax = axes[1,0]
ax.text(0.03, 0.90, txt, transform=ax.transAxes)

# Set axis limits and ticks
r_min = 9
r_max = 14
m_min = 1.0
m_max = 2.6
xticks = [9, 10, 11, 12, 13, 14]
yticks = [m_min, 1.4, 1.8, 2.2]
[ax.set_xlim(r_min,r_max) for ax in axes.flatten()]
[ax.set_ylim(m_min,m_max) for ax in axes.flatten()]

[ax.set_xticks(xticks[:-1]) for ax in axes[:,0]]
[ax.set_xticks(xticks) for ax in axes[:,1]]
axes[1,0].set_xticks(xticks[:-1])
axes[1,1].set_xticks(xticks)

[ax.set_yticks(yticks) for ax in axes.flatten()]
axes[0,0].set_yticks(yticks + [2.6])

[ax.set_xticklabels([]) for ax in axes[0,:]]
[ax.set_yticklabels([]) for ax in axes[:,1]]

# Minor ticks
[ax.minorticks_on() for ax in axes.flatten()]

# Ticks on right and top side
[plotting.right_side_ticks(ax) for ax in axes[:,1]]
[plotting.top_side_ticks(ax) for ax in axes[0,:]]

# Z-order the panels. Needed for the ticks to show up properly
# Right on top of left, top on top of bottom.
axes[1,0].set_zorder(1)
axes[1,1].set_zorder(2)
axes[0,0].set_zorder(3)
axes[0,1].set_zorder(4)

# Add legends
line1 = plotting.custom_line(c_n2lo[1], lw=1)
line2 = plotting.double_interval_legend(*c_n3lo)
custom_lines = [line1, line2]
labels = [r'$\chi$EFT $\leq 1.1 n_0$', r'$\chi$EFT $\leq 1.5 n_0$']
loc = 'lower left'
axes[0,0].legend(custom_lines, labels, loc=loc, frameon=False)

# Save result
plotting.save(fig, f'{parent}/fig', f'MR_baseline_new.pdf', args.tmp_output)
