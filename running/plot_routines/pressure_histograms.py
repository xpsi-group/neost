# Stardard libraries
import argparse
import os
import pathlib
import sys

# 3rd party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Local
import plotting

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tmp_output', action='store_true')
parser.add_argument('-o', '--order', type=str)
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

order = 'N3LO' if args.order is None else args.order
order_lower = order.lower()

def plot(ax, y, colors, priors=False, face_alpha=None):
    lw = 0.5
    if priors is True:
        ax.hist(np.log10(y[y>0.]), bins=20, density=True, histtype='step', linestyle='-', color=colors, alpha=0.8, lw=lw)
    else:
        ax.hist(np.log10(y[y>0]), bins=20, density=True, histtype='stepfilled', facecolor=colors, alpha=face_alpha)
        ax.hist(np.log10(y[y>0]), bins=20, density=True,  histtype='step',color=colors, alpha=1.0, lw=0.4)

# Paths
current_path = os.path.dirname(__file__)
parent = os.path.dirname(current_path)

# Create subplots, remove spaces
width = 7
height = 5.4
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(width,height))
fig.subplots_adjust(wspace=0, hspace=0.15)

face_alpha = 0.9

# Set axis limits and ticks
[ax.set_yticks([]) for ax in axes.flatten()]
xticks1 = [34.0, 34.2, 34.4, 34.6, 34.8, 35.0]
xticks2 = [34.2, 34.4, 34.6, 34.8, 35.0, 35.2, 35.4]
axes[0,0].set_xticks(xticks1[:-1])
axes[0,1].set_xticks(xticks1)
axes[1,0].set_xticks(xticks2[:-1])
axes[1,1].set_xticks(xticks2)
axes[0,0].set_xticklabels(xticks1[:-1])
axes[0,1].set_xticklabels(xticks1)
axes[1,0].set_xticklabels(xticks2[:-1])
axes[1,1].set_xticklabels(xticks2)

xlim1 = (34.0, 35.0)
xlim2 = (34.2, 35.4)
[ax.set_xlim(xlim1) for ax in axes[0,:]]
[ax.set_xlim(xlim2) for ax in axes[1,:]]
ymax = 8.0
axes[0,0].set_ylim(0,ymax)

# Set labels and titles
ylabel = 'PDF'
xlabel = r'$\log_{10}(P)$ [dyn/cm$^2$]'
axes[1,0].set_xlabel(xlabel)
axes[1,1].set_xlabel(xlabel)
axes[0,0].set_ylabel(ylabel)
axes[1,0].set_ylabel(ylabel)
axes[0,0].set_title(f'PP')
axes[0,1].set_title(f'CS')

# Define colors
c_n3lo = plotting.c_n3lo
c_n2lo = plotting.c_n2lo
c_hebeler = plotting.c_hebeler
c_n3lo_11 = plotting.c_n3lo_11

# Results directory
directory = f'{parent}/results' if not args.repro else f'{parent}/repro'

##################################### fix this
posterior_path = f'{directory}/posterior'
prior_path = f'{directory}/prior'

pressure_11_2_pp = np.load(f'{posterior_path}/pp/{order_lower}/11/new/press_n20.npy')
pressure_11_2_cs = np.load(f'{posterior_path}/cs/{order_lower}/11/new/press_n20.npy')

pressure_15_2_pp = np.load(f'{posterior_path}/pp/{order_lower}/15/new/press_n20.npy')
pressure_15_2_cs = np.load(f'{posterior_path}/cs/{order_lower}/15/new/press_n20.npy')

pressure_11_3_pp = np.load(f'{posterior_path}/pp/{order_lower}/11/new/press_n30.npy')
pressure_11_3_cs = np.load(f'{posterior_path}/cs/{order_lower}/11/new/press_n30.npy')

pressure_15_3_pp = np.load(f'{posterior_path}/pp/{order_lower}/15/new/press_n30.npy')
pressure_15_3_cs = np.load(f'{posterior_path}/cs/{order_lower}/15/new/press_n30.npy')

prior_pressure_11_2_pp = np.load(f'{prior_path}/pp/{order_lower}/11/press_n20.npy')
prior_pressure_11_2_cs = np.load(f'{prior_path}/cs/{order_lower}/11/press_n20.npy')

prior_pressure_15_2_pp = np.load(f'{prior_path}/pp/{order_lower}/15/press_n20.npy')
prior_pressure_15_2_cs = np.load(f'{prior_path}/cs/{order_lower}/15/press_n20.npy')

prior_pressure_11_3_pp = np.load(f'{prior_path}/pp/{order_lower}/11/press_n30.npy')
prior_pressure_11_3_cs = np.load(f'{prior_path}/cs/{order_lower}/11/press_n30.npy')

prior_pressure_15_3_pp = np.load(f'{prior_path}/pp/{order_lower}/15/press_n30.npy')
prior_pressure_15_3_cs = np.load(f'{prior_path}/cs/{order_lower}/15/press_n30.npy')
############################################

# PP
ax = axes[0,0]
plot(ax, pressure_11_2_pp, c_n2lo[0], face_alpha=face_alpha)
plot(ax, prior_pressure_11_2_pp, c_n2lo[1], True, face_alpha=face_alpha)

plot(ax, pressure_15_2_pp, c_n3lo[0], face_alpha=face_alpha)
plot(ax, prior_pressure_15_2_pp, c_n3lo[1], True, face_alpha=face_alpha)

ax = axes[1,0]
plot(ax, pressure_11_3_pp, c_n2lo[0], face_alpha=face_alpha)
plot(ax, prior_pressure_11_3_pp, c_n2lo[1], True, face_alpha=face_alpha)

plot(ax, pressure_15_3_pp, c_n3lo[0], face_alpha=face_alpha)
plot(ax, prior_pressure_15_3_pp, c_n3lo[1], True, face_alpha=face_alpha)

# CS
ax = axes[0,1]
plot(ax, pressure_11_2_cs, c_n2lo[0], face_alpha=face_alpha)
plot(ax, prior_pressure_11_2_cs, c_n2lo[1], True, face_alpha=face_alpha)

plot(ax, pressure_15_2_cs, c_n3lo[0], face_alpha=face_alpha)
plot(ax, prior_pressure_15_2_cs, c_n3lo[1], True, face_alpha=face_alpha)

ax = axes[1,1]
plot(ax, pressure_11_3_cs, c_n2lo[0], face_alpha=face_alpha)
plot(ax, prior_pressure_11_3_cs, c_n2lo[1], True, face_alpha=face_alpha)

plot(ax, pressure_15_3_cs, c_n3lo[0], face_alpha=face_alpha)
plot(ax, prior_pressure_15_3_cs, c_n3lo[1], True, face_alpha=face_alpha)

# Add text
ax = axes[0,0]
x = 0.04
y = 0.89
y2 = 0.79
txt = f'$n = 2 n_0$'
order_string = r'N$^2$LO' if order == 'N2LO' else r'N$^3$LO'
axes[0,0].text(x, y, txt, transform=axes[0,0].transAxes)
axes[0,0].text(x, y2, order_string, transform=axes[0,0].transAxes)
txt = f'$n = 3 n_0$'
axes[1,0].text(x, y, txt, transform=axes[1,0].transAxes)
axes[1,0].text(x, y2, order_string, transform=axes[1,0].transAxes)

# Minor ticks
[ax.minorticks_on() for ax in axes.flatten()]

# Z-order the panels. Needed for the ticks to show up properly
# Right on top of left, top on top of bottom.
axes[1,0].set_zorder(1)
axes[1,1].set_zorder(2)
axes[0,0].set_zorder(3)
axes[0,1].set_zorder(4)

# Add legends
line0 = plotting.custom_line('xkcd:black', '-', lw=0.6)
line2 = plotting.custom_patch(c_n2lo[0], alpha=face_alpha)
line3 = plotting.custom_patch(c_n3lo[0], alpha=face_alpha)
custom_lines = [line0, line2, line3]
labels = ['Priors', r'$\chi$EFT $\leq 1.1 n_0$', r'$\chi$EFT $\leq 1.5 n_0$']
loc = (-0.15,0.65)
axes[0,1].legend(custom_lines, labels, loc=loc, frameon=False)

custom_lines = [line0, line2, line3]
axes[1,1].legend(custom_lines, labels, loc=loc, frameon=False)

# Save result
string = '_N2LO' if order == 'N2LO' else ''
plotting.save(fig, f'{parent}/fig', f'pressure_new_histograms{string}.pdf', args.tmp_output)
