# Standard libraries
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
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(width,height))
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
c_n3lo = plotting.c_n3lo
c_n2lo = plotting.c_n2lo
c_hebeler = plotting.c_hebeler
c_n3lo_11 = plotting.c_n3lo_11

# Results directory
directory = f'{parent}/results/prior' if not args.repro else f'{parent}/repro/prior'

# PP
# Keller N2LO, 1.1*rho_ns
pp_n2lo_11 = np.loadtxt(f'{directory}/pp/n2lo/11/MR_prpr.txt')
# Keller N2LO, 1.5*rho_ns
pp_n2lo_15 = np.loadtxt(f'{directory}/pp/n2lo/15/MR_prpr.txt')
# Keller N3LO, 1.1*rho_ns
pp_n3lo_11 = np.loadtxt(f'{directory}/pp/n3lo/11/MR_prpr.txt')
# Keller N3LO, 1.5*rho_ns
pp_n3lo_15 = np.loadtxt(f'{directory}/pp/n3lo/15/MR_prpr.txt')
# Hebeler
pp_hebeler = np.loadtxt(f'{directory}/pp/hebeler/11/MR_prpr.txt')

# CS
# Keller N2LO, 1.1*rho_ns
cs_n2lo_11 = np.loadtxt(f'{directory}/cs/n2lo/11/MR_prpr.txt')
# Keller N2LO, 1.5*rho_ns
cs_n2lo_15 = np.loadtxt(f'{directory}/cs/n2lo/15/MR_prpr.txt')
# Keller N3LO, 1.1*rho_ns
cs_n3lo_11 = np.loadtxt(f'{directory}/cs/n3lo/11/MR_prpr.txt')
# Keller N3LO, 1.5*rho_ns
cs_n3lo_15 = np.loadtxt(f'{directory}/cs/n3lo/15/MR_prpr.txt')
# Hebeler
cs_hebeler = np.loadtxt(f'{directory}/cs/hebeler/11/MR_prpr.txt')

# PP
ax = axes[0,0]
plot(ax, pp_n3lo_11, c_n3lo, True)
plot(ax, pp_n2lo_11, c_n2lo, False, '-')
plot(ax, pp_hebeler, c_hebeler, False, 'dotted')

ax = axes[1,0]
plot(ax, pp_n3lo_15, c_n3lo, True)
plot(ax, pp_n2lo_15, c_n2lo, False, '-')

# CS
ax = axes[0,1]
plot(ax, cs_n3lo_11, c_n3lo, True)
plot(ax, cs_n2lo_11, c_n2lo, False, '-')
plot(ax, cs_hebeler, c_hebeler, False, 'dotted')

ax = axes[1,1]
plot(ax, cs_n3lo_15, c_n3lo, True)
plot(ax, cs_n2lo_15, c_n2lo, False, '-')

# Add text
txt = r'$\chi$EFT $\leq 1.1 n_0$'
ax = axes[0,0]
ax.text(0.03, 0.90, txt, transform=ax.transAxes)
txt = r'$\chi$EFT $\leq 1.5 n_0$'
ax = axes[1,0]
ax.text(0.03, 0.90, txt, transform=ax.transAxes)

# Set axis limits and ticks
r_min = 6
r_max = 15
m_min = 1.0
m_max = 2.6
xticks = [8, 10, 12, 14]
yticks = [m_min, 1.4, 1.8, 2.2]
axes[0,0].set_yticks(yticks)
[ax.set_xlim(r_min,r_max) for ax in axes.flatten()]
[ax.set_ylim(m_min,m_max) for ax in axes.flatten()]
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
axes[1,0].set_zorder(1)
axes[1,1].set_zorder(2)
axes[0,0].set_zorder(3)
axes[0,1].set_zorder(4)

# Add legends
legend_hebeler = plotting.custom_line(c_hebeler[1], 'dotted', lw=1)
legend_n2lo = plotting.custom_line(c_n2lo[1], lw=1)
legend_n3lo = plotting.double_interval_legend(*c_n3lo)

custom_lines = [legend_hebeler, legend_n2lo, legend_n3lo]
labels = ['Hebeler et al.', 'N$^2$LO', 'N$^3$LO']
axes[0,1].legend(custom_lines, labels, loc=(0.02,0.66), frameon=False)

custom_lines = [legend_n2lo, legend_n3lo]
labels = ['N$^2$LO', 'N$^3$LO']
axes[1,1].legend(custom_lines, labels, loc=(0.02,0.76), frameon=False)

# Save result
plotting.save(fig, f'{parent}/fig', 'MR_priors.pdf', args.tmp_output)
