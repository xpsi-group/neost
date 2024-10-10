# Stardard libraries
import argparse
import os

# 3rd party
import matplotlib.pyplot as plt
import numpy as np

# Local
import plotting

# Paths
current_path = os.path.dirname(__file__)
parent = os.path.dirname(current_path)

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tmp_output', action='store_true')
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

# Plotting function
plot = plotting.plot_1D_quantiles

# Create subplots, remove spaces
width = 7
height = 5
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(width,height))
fig.subplots_adjust(wspace=0, hspace=0)

# Set axis limits and ticks
[ax.minorticks_on() for ax in axes.flatten()]
[ax.set_xticklabels([]) for ax in axes.flatten()]
[ax.set_yticklabels([]) for ax in axes.flatten()]
[ax.set_xticks([14.2, 14.6, 15.0]) for ax in axes[:,0]]
[ax.set_xticks([14.2, 14.6, 15.0, 15.4]) for ax in axes[:,1]]
axes[1,0].set_xticklabels([14.2, 14.6, 15.0])
axes[1,1].set_xticklabels([14.2, 14.6, 15.0, 15.4])
[ax.set_xlim(14.2,15.4) for ax in axes.flatten()]
[ax.set_ylim(32.8,36.4) for ax in axes.flatten()]

[ax.set_yticks([33,34,35,36]) for ax in axes.flatten()]
[ax.set_yticklabels([33,34,35,36]) for ax in axes[:,0]]

# Set labels and titles
xlabel = r'$\log_{10}(\varepsilon)$ [g/cm$^3$]'
ylabel = r'$\log_{10}(P)$ [dyn/cm$^2$]'
axes[1,0].set_xlabel(xlabel)
axes[1,1].set_xlabel(xlabel)
axes[0,0].set_ylabel(ylabel)
axes[1,0].set_ylabel(ylabel)
axes[0,0].set_title('PP')
axes[0,1].set_title('CS')

# Define colors
c_n3lo = plotting.c_n3lo
c_n2lo = plotting.c_n2lo
c_hebeler = plotting.c_hebeler

# Results directory
directory = f'{parent}/results/prior' if not args.repro else f'{parent}/repro/prior'

# PP
# Keller N2LO, 1.1*rho_ns
pp_n2lo_11 = np.load(f'{directory}/pp/n2lo/11/pressures.npy')
# Keller N2LO, 1.5*rho_ns
pp_n2lo_15 = np.load(f'{directory}/pp/n2lo/15/pressures.npy')
# Keller N3LO, 1.1*rho_ns
pp_n3lo_11 = np.load(f'{directory}/pp/n3lo/11/pressures.npy')
# Keller N3LO, 1.5*rho_ns
pp_n3lo_15 = np.load(f'{directory}/pp/n3lo/15/pressures.npy')
# Hebeler
pp_hebeler = np.load(f'{directory}/pp/hebeler/11/pressures.npy')

# CS
# Keller N2LO, 1.1*rho_ns
cs_n2lo_11 = np.load(f'{directory}/cs/n2lo/11/pressures.npy')
# Keller N2LO, 1.5*rho_ns
cs_n2lo_15 = np.load(f'{directory}/cs/n2lo/15/pressures.npy')
# Keller N3LO, 1.1*rho_ns
cs_n3lo_11 = np.load(f'{directory}/cs/n3lo/11/pressures.npy')
# Keller N3LO, 1.5*rho_ns
cs_n3lo_15 = np.load(f'{directory}/cs/n3lo/15/pressures.npy')
# Hebeler
cs_hebeler = np.load(f'{directory}/cs/hebeler/11/pressures.npy')

# PP 1.1
ax = axes[0,0]
plot(ax, pp_n3lo_11, c_n3lo, 'filled')
plot(ax, pp_n2lo_11, c_n2lo, '-')
plot(ax, pp_hebeler, c_hebeler, 'dotted')

# PP 1.5
ax = axes[1,0]
plot(ax, pp_n3lo_15, c_n3lo, 'filled')
plot(ax, pp_n2lo_15, c_n2lo, '-')

# CS 1.1
ax = axes[0,1]
plot(ax, cs_n3lo_11, c_n3lo, 'filled')
plot(ax, cs_n2lo_11, c_n2lo, '-')
plot(ax, cs_hebeler, c_hebeler, 'dotted')

# CS 1.5
ax = axes[1,1]
plot(ax, cs_n3lo_15, c_n3lo, 'filled')
plot(ax, cs_n2lo_15, c_n2lo, '-')

# Add x*n_0
n0 = 2.68e14 # n0 in g/cm^3
[ax.axvline(np.log10(1.1*n0), color='xkcd:gray', lw=0.5) for ax in axes[0,:]]
[ax.axvline(np.log10(1.5*n0), color='xkcd:gray', lw=0.5) for ax in axes[1,:]]

# Add text
ax = axes[0,0]
txt = r'$\chi$EFT $\leq 1.1 n_0$'
ax.text(0.03, 0.90, txt, transform=ax.transAxes)

ax = axes[1,0]
txt = r'$\chi$EFT $\leq 1.5 n_0$'
ax.text(0.03, 0.90, txt, transform=ax.transAxes)

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
legend_hebeler = plotting.custom_line(c_hebeler[0], 'dotted', lw=1)
legend_n2lo = plotting.custom_line(c_n2lo[1], lw=1)
legend_n3lo = plotting.double_interval_legend(*c_n3lo)
custom_lines = [legend_hebeler, legend_n2lo, legend_n3lo]
labels = ['Hebeler et al.', 'N$^2$LO', 'N$^3$LO']
axes[0,1].legend(custom_lines, labels, loc=(0.02,0.66), frameon=False)

custom_lines = [legend_n2lo, legend_n3lo]
labels = ['N$^2$LO', 'N$^3$LO']
axes[1,1].legend(custom_lines, labels, loc=(0.02,0.76), frameon=False)

# Save result
plotting.save(fig, f'{parent}/fig', 'P_eps_priors.pdf', args.tmp_output)
