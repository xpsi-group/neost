# Standard libraries
import argparse
import os

# 3rd party
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Local
import plotting

def stacking(data, order, density, param):
    x1 = data[f'{order}_{density}{param}_Mmax'].reshape((-1,1))
    x2 = data[f'{order}_{density}{param}_delta_R'].reshape((-1,1))
    return np.hstack((x1, x2))

# Plot function
plot = plotting.plot_KDE

# Path variables
current_path = os.path.dirname(__file__)
parent = os.path.dirname(current_path)

# Read input
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument('-bo', '--both_orders', action='store_true')
parser.add_argument('-t', '--tmp_output', action='store_true')
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

onlyn3lo = False if args.both_orders else True

dataset = args.dataset.lower()
dataset = ''.join(dataset.split())
assert(dataset in ['baseline', 'new', 'new2', 'new3'])
dataset_nice = 'Baseline' if dataset == 'baseline' else f'New {dataset[-1]}'

# Results directory
directory = f'{parent}/results/posterior' if not args.repro else f'{parent}/repro/posterior'

data = {}
orders = ['N3LO'] if onlyn3lo else ['N2LO', 'N3LO']
for param in ['pp', 'cs']:
    for density in ['11', '15']:
        for order in orders:
            fname = f'{directory}/{param}/{order.lower()}/{density}/{dataset}/table_data.txt'
            file = np.loadtxt(fname)
            r14 = file[:,5]
            r2 = file[:,9]
            delta_R = r2 - r14
            key = f'{order}_{density}{param}_Mmax'
            data[key] = file[:, 0]
            key = f'{order}_{density}{param}_delta_R'
            data[key] = delta_R

# Set up figure
width = 6
height = 3 if onlyn3lo else 6

nrows = 1 if onlyn3lo else 2
fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=False, sharey=False, figsize=(width,height))
fig.subplots_adjust(wspace=0, hspace=0)
if onlyn3lo:
    axes = axes.reshape((1, 2))

# Some labels
xlabel = r'$\Delta R$ [km]'
ylabel = r'$M_\textnormal{TOV}$ $[M_{\odot}]$'
[ax.set_xlabel(xlabel) for ax in axes[-1,:]]
[ax.set_ylabel(ylabel) for ax in axes[:,0]]
axes[0,0].set_title('PP')
axes[0,1].set_title('CS')

# Set axis limits and ticks
delta_R_min = -2.0
delta_R_max = 0.60
maxmass_min = 1.95
maxmass_max = 2.5

xticks = [-2, -1, 0, 1]
xticklabels = ['$-2$', '$-1$', '$0$', '$1$']
yticks = [2.0, 2.2, 2.4, 2.6]

[ax.set_xticklabels([]) for ax in axes.flatten()]
[ax.set_yticklabels([]) for ax in axes.flatten()]
[ax.set_xticks(xticks) for ax in axes.flatten()]
[ax.set_xticklabels(xticklabels) for ax in axes[-1,:]]
[ax.set_yticks(yticks) for ax in axes.flatten()]
[ax.set_yticklabels(yticks) for ax in axes[:,0]]
[ax.minorticks_on() for ax in axes.flatten()]
[plotting.right_side_ticks(ax) for ax in axes[:,1]]
[plotting.top_side_ticks(ax) for ax in axes[0,:]]


c_n3lo = plotting.c_n3lo_posterior
c_n2lo = plotting.c_n2lo

# Plotting
if onlyn3lo:
    # PP
    ax = axes[0,0]
    x = stacking(data, 'N3LO', '15', 'pp')
    plot(ax, x, c_n3lo, True)

    x = stacking(data, 'N3LO', '11', 'pp')
    plot(ax, x, c_n2lo, False)

    # CS
    ax = axes[0,1]
    x = stacking(data, 'N3LO', '15', 'cs')
    plot(ax, x, c_n3lo, True)

    x = stacking(data, 'N3LO', '11', 'cs')
    plot(ax, x, c_n2lo, False)
else:
    # PP
    ax = axes[0,0]
    x = stacking(data, 'N3LO', '11', 'pp')
    plot(ax, x, c_n3lo, True)

    x = stacking(data, 'N2LO', '11', 'pp')
    plot(ax, x, c_n2lo, False)

    ax = axes[1,0]
    x = stacking(data, 'N3LO', '15', 'pp')
    plot(ax, x, c_n3lo, True)

    x = stacking(data, 'N2LO', '15', 'pp')
    plot(ax, x, c_n2lo, False)

    # CS
    ax = axes[0,1]
    x = stacking(data, 'N3LO', '11', 'cs')
    plot(ax, x, c_n3lo, True)

    x = stacking(data, 'N2LO', '11', 'cs')
    plot(ax, x, c_n2lo, False)

    ax = axes[1,1]
    x = stacking(data, 'N3LO', '15', 'cs')
    plot(ax, x, c_n3lo, True)

    x = stacking(data, 'N2LO', '15', 'cs')
    plot(ax, x, c_n2lo, False)

# Add text
if onlyn3lo:
    ax = axes[0,0]
    txt = 'N$^3$LO'
    ax.text(0.04, 0.9, txt, transform=ax.transAxes)
else:
    ax = axes[0,0]
    txt = r'$\chi$EFT $\leq 1.1 n_0$'
    ax.text(0.04, 0.90, txt, transform=ax.transAxes)

    ax = axes[1,0]
    txt = r'$\chi$EFT $\leq 1.5 n_0$'
    ax.text(0.04, 0.90, txt, transform=ax.transAxes)

[ax.set_zorder(i) for i, ax in enumerate(axes.flatten())]

# Add legends
line0 = plotting.custom_line(c_n2lo[1], lw=1)
line1 = plotting.double_interval_legend(*c_n3lo)
custom_lines = [line0, line1]
labels = None
if onlyn3lo:
    labels = [r'$\chi$EFT $\leq 1.1 n_0$', r'$\chi$EFT $\leq 1.5 n_0$']
else:
    labels = ['N$^2$LO', 'N$^3$LO']
[ax.legend(custom_lines, labels, loc='upper left', frameon=False) for ax in axes[:,1]]

[ax.set_xlim(delta_R_min, delta_R_max) for ax in axes.flatten()]
[ax.set_ylim(maxmass_min, maxmass_max) for ax in axes.flatten()]

# Z-order the panels. Needed for the ticks to show up properly
# Right on top of left, top on top of bottom.
tmp = axes[:, [1,0]]
tmp = np.flip(tmp.flatten())
[ax.set_zorder(i) for i, ax in enumerate(tmp)]

fname = f'deltaR_MaxMass_{dataset}.pdf' if onlyn3lo else f'deltaR_MaxMass_{dataset}_N2LO_N3LO.pdf'
plotting.save(fig, f'{parent}/fig', fname, args.tmp_output)
