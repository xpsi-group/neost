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

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--core_start', required=True, type=int)
parser.add_argument('-o', '--order', type=str)
parser.add_argument('-t', '--tmp_output', action='store_true')
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

# Order and core start
order = 'N3LO' if args.order is None else args.order
order_lower = order.lower()
core_start = args.core_start

# Create subplots, remove spaces
width = 7
height = 7.5
fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(width,height))
fig.subplots_adjust(wspace=0, hspace=0)

# Plotting function
plot = plotting.plot_KDE

# Set labels and titles
xlabel = '$R$ [km]'
ylabel = r'$M$ [$M_{\odot}$]'
axes[2,0].set_xlabel(xlabel)
axes[2,1].set_xlabel(xlabel)
axes[0,0].set_ylabel(ylabel)
axes[1,0].set_ylabel(ylabel)
axes[2,0].set_ylabel(ylabel)
axes[0,0].set_title('PP')
axes[0,1].set_title('CS')

# Define colors
c_n3lo = plotting.c_n3lo_posterior
c_n2lo = plotting.c_n2lo

# Results directory
directory = f'{parent}/results/posterior' if not args.repro else f'{parent}/repro/posterior'

# PP
pp_baseline = np.loadtxt(f'{directory}/pp/{order_lower}/{core_start}/baseline/MR_prpr.txt')
pp_new = np.loadtxt(f'{directory}/pp/{order_lower}/{core_start}/new/MR_prpr.txt')
pp_new2 = np.loadtxt(f'{directory}/pp/{order_lower}/{core_start}/new2/MR_prpr.txt')
pp_new3 = np.loadtxt(f'{directory}/pp/{order_lower}/{core_start}/new3/MR_prpr.txt')

# CS
cs_baseline = np.loadtxt(f'{directory}/cs/{order_lower}/{core_start}/baseline/MR_prpr.txt')
cs_new = np.loadtxt(f'{directory}/cs/{order_lower}/{core_start}/new/MR_prpr.txt')
cs_new2 = np.loadtxt(f'{directory}/cs/{order_lower}/{core_start}/new2/MR_prpr.txt')
cs_new3 = np.loadtxt(f'{directory}/cs/{order_lower}/{core_start}/new3/MR_prpr.txt')

# PP
ax = axes[0,0]
plot(ax, pp_new, c_n3lo, True)
plot(ax, pp_baseline, c_n2lo, False, '-')

ax = axes[1,0]
plot(ax, pp_new2, c_n3lo, True)
plot(ax, pp_baseline, c_n2lo, False, '-')

ax = axes[2,0]
plot(ax, pp_new3, c_n3lo, True)
plot(ax, pp_baseline, c_n2lo, False, '-')

# CS
ax = axes[0,1]
plot(ax, cs_new, c_n3lo, True)
plot(ax, cs_baseline, c_n2lo, False, '-')

ax = axes[1,1]
plot(ax, cs_new2, c_n3lo, True)
plot(ax, cs_baseline, c_n2lo, False, '-')

ax = axes[2,1]
plot(ax, cs_new3, c_n3lo, True)
plot(ax, cs_baseline, c_n2lo, False, '-')

# Add text
txt1 = 'N$^3$LO'
txt2 = rf'$\chi$EFT $\leq {core_start/10:.1f} n_0$'
ax = axes[2,0]
ax.text(0.04, 0.14, txt1, transform=ax.transAxes)
ax.text(0.04, 0.05, txt2, transform=ax.transAxes)
#txt = "Chiral EFT $\leq 1.5 n_0$"
#ax = axes[1,0]
#ax.text(0.03, 0.90, txt, transform=ax.transAxes)

# Set axis limits and ticks
r_min = 10
r_max = 14
m_min = 1.0
m_max = 2.6
xticks = [10, 11, 12, 13, 14]
yticks = [m_min, 1.4, 1.8, 2.2]

[ax.set_xlim(r_min,r_max) for ax in axes.flatten()]
[ax.set_ylim(m_min,m_max) for ax in axes.flatten()]

[ax.set_xticks(xticks[:-1]) for ax in axes[:,0]]
[ax.set_xticks(xticks) for ax in axes[:,1]]
[ax.set_xticklabels([]) for ax in axes.flatten()]
axes[2,0].set_xticklabels(xticks[:-1])
axes[2,1].set_xticklabels(xticks)

[ax.set_yticks(yticks) for ax in axes.flatten()]
axes[0,0].set_yticks(yticks + [2.6])
[ax.set_yticklabels([]) for ax in axes[:,1]]

# Minor ticks
[ax.minorticks_on() for ax in axes.flatten()]

# Ticks on right and top side
[plotting.right_side_ticks(ax) for ax in axes[:,1]]
[plotting.top_side_ticks(ax) for ax in axes[0,:]]


# Z-order the panels. Needed for the ticks to show up properly
# Right on top of left, top on top of bottom.
axes[2,0].set_zorder(1)
axes[2,1].set_zorder(2)
axes[1,0].set_zorder(3)
axes[1,1].set_zorder(4)
axes[0,0].set_zorder(5)
axes[0,1].set_zorder(6)

# Add legends
line1 = plotting.custom_line(c_n2lo[1])
line2 = plotting.double_interval_legend(*c_n3lo)
custom_lines = [line1, line2]
loc = 'upper left'
axes[0,0].legend(custom_lines, ['Baseline', 'New'], loc=loc, frameon=False)
axes[1,0].legend(custom_lines, ['Baseline', 'New 2'], loc=loc, frameon=False)
axes[2,0].legend(custom_lines, ['Baseline', 'New 3'], loc=loc, frameon=False)

plotting.save(fig, f'{parent}/fig', f'MR_{order}_{core_start}_comparison.pdf', args.tmp_output)
