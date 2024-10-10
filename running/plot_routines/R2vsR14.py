# Standard libraries
import argparse
import os

# Third party
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Local
import plotting


def stacking(y, x):
    return np.hstack((y.reshape(-1,1), x.reshape(-1,1)))
    
plot = plotting.plot_KDE

# Paths
current_path = os.path.dirname(__file__)
parent = os.path.dirname(current_path)

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tmp_output', action='store_true')
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

# Results directory
directory = f'{parent}/results/posterior' if not args.repro else f'{parent}/repro/posterior'

# PP
# Keller N3LO, 1.1*rho_ns
fname = f'{directory}/pp/n3lo/11/new/table_data.txt'
data = np.loadtxt(fname)
pp_n3lo_11_r14 = data[:,5]
pp_n3lo_11_r2 = data[:,9]

# Keller N3LO, 1.5*rho_ns
fname = f'{directory}/pp/n3lo/15/new/table_data.txt'
data = np.loadtxt(fname)
pp_n3lo_15_r14 = data[:,5]
pp_n3lo_15_r2 = data[:,9]

# CS
# Keller N3LO, 1.1*rho_ns
fname = f'{directory}/cs/n3lo/11/new/table_data.txt'
data = np.loadtxt(fname)
cs_n3lo_11_r14 = data[:,5]
cs_n3lo_11_r2 = data[:,9]

# Keller N3LO, 1.5*rho_ns
fname = f'{directory}/cs/n3lo/15/new/table_data.txt'
data = np.loadtxt(fname)
cs_n3lo_15_r14 = data[:,5]
cs_n3lo_15_r2 = data[:,9]

width = 6
height = 3

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(width,height))
fig.subplots_adjust(wspace=0, hspace=0)

# Set axis limits and ticks
r_min = 10
r_max = 14
r14_min = r_min
r14_max = r_max
r2_min = r_min
r2_max = r_max
ticks = [10,11,12,13,14]
xlabel = '$R_{1.4}$ [km]'
ylabel = '$R_{2.0}$ [km]'
axes[0].set_xlabel(xlabel)
axes[1].set_xlabel(xlabel)
axes[0].set_ylabel(ylabel)
axes[0].set_title('PP')
axes[1].set_title('CS')

[ax.minorticks_on() for ax in axes.flatten()]

[ax.set_xticklabels([]) for ax in axes.flatten()]
[ax.set_yticklabels([]) for ax in axes.flatten()]

axes[0].set_xticks(ticks[:-1])
axes[0].set_xticklabels(ticks[:-1])
axes[1].set_xticks(ticks)
axes[1].set_xticklabels(ticks)

[ax.set_yticks(ticks) for ax in axes.flatten()]
axes[0].set_yticklabels(ticks)

# This is a somewhat convoluted way of getting the right minorticks on the top of axes[0].
# It's unclear to me why this is necessary for the top ticks but not the bottom ticks.
minorticks = axes[1].get_xticks(minor=True)
axes[0].set_xticks(minorticks, minor=True)

# Top and right side ticks
plotting.right_side_ticks(axes[1])
[plotting.top_side_ticks(ax) for ax in axes]

# Colors
c_n3lo = plotting.c_n3lo_posterior
c_n2lo = plotting.c_n2lo
c_hebeler = plotting.c_hebeler
c_n3lo_11 = plotting.c_n3lo_11

diagonal = np.linspace(r_min, r_max)

bw_adjust = 2.0 # bw_adjust taken to 2 because contours seemed jagged and had some weird kinks

# PP
ax = axes[0]
data = stacking(pp_n3lo_15_r2, pp_n3lo_15_r14)
plot(ax, data, c_n3lo, True, bw_adjust=bw_adjust)
data = stacking(pp_n3lo_11_r2, pp_n3lo_11_r14)
plot(ax, data, c_n2lo, False, '-', bw_adjust=bw_adjust)
ax.plot(diagonal, diagonal, color='xkcd:gray', lw=0.5)

# CS
ax = axes[1]
data = stacking(cs_n3lo_15_r2, cs_n3lo_15_r14)
plot(ax, data, c_n3lo, True, bw_adjust=bw_adjust)
data = stacking(cs_n3lo_11_r2, cs_n3lo_11_r14)
plot(ax, data, c_n2lo, False, '-', bw_adjust=bw_adjust)
ax.plot(diagonal, diagonal, color='xkcd:gray', lw=0.5)

axes[0].set_zorder(1)
axes[1].set_zorder(2)

# Add legends
line1 = plotting.custom_line(c_n2lo[1], lw=1)
line2 = plotting.double_interval_legend(*c_n3lo)
custom_lines = [line1, line2]
loc = 'upper left'
axes[1].legend(custom_lines, [r"$\chi$EFT $\leq 1.1 n_0$", r"$\chi$EFT $\leq 1.5 n_0$"], loc=loc, frameon=False)

# Add text
ax = axes[0]
txt = 'N$^3$LO'
ax.text(0.04, 0.90, txt, transform=ax.transAxes)

# Axis limits
[ax.set_xlim(r14_min,r14_max) for ax in axes.flatten()]
[ax.set_ylim(r2_min,r2_max) for ax in axes.flatten()]

# Save result
plotting.save(fig, f'{parent}/fig', 'R2_vs_R14_new.pdf', args.tmp_output)
