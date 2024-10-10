# Standard libraries
import argparse
import os

# Third party
import matplotlib.pyplot as plt
import numpy as np

# Local
import plotting

# Paths
current_path = os.path.dirname(__file__)
parent = os.path.dirname(current_path)
data_path = f'{parent}/data/astro_data'

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tmp_output', action='store_true')
parser.add_argument('-a', '--all', action='store_true') # If set, plot Baseline + New 1, 2 and 3. Otherwise just Baseline and New (2).
args = parser.parse_args()
all_scenarios = True if args.all else False

# Read in MR data data from files
# Column 1 = Radius in km
# Column 2 = Mass in solar mass

# J0030 ST+PST (Riley et al 2019)

J0030_R19_STPST_MR_95 = np.loadtxt(f'{data_path}/J0030_Riley19_NICERonly_STPST_95.txt')
J0030_R19_STPST_M_95 = J0030_R19_STPST_MR_95[:,1]
J0030_R19_STPST_R_95 = J0030_R19_STPST_MR_95[:,0]

J0030_R19_STPST_MR_68 = np.loadtxt(f'{data_path}/J0030_Riley19_NICERonly_STPST_68.txt')
J0030_R19_STPST_M_68 = J0030_R19_STPST_MR_68[:,1]
J0030_R19_STPST_R_68 = J0030_R19_STPST_MR_68[:,0]


# J0030 NICER only ST+PST (Vinciguerra et al. 2024)

J0030_V24_STPST_MR_95  = np.loadtxt(f'{data_path}/J0030_Vinciguerra24_STPST_NICERonly_95.txt')
J0030_V24_STPST_M_95  = J0030_V24_STPST_MR_95[:,1]
J0030_V24_STPST_R_95  = J0030_V24_STPST_MR_95[:,0]

J0030_V24_STPST_MR_68  = np.loadtxt(f'{data_path}/J0030_Vinciguerra24_STPST_NICERonly_68.txt')
J0030_V24_STPST_M_68  = J0030_V24_STPST_MR_68[:,1]
J0030_V24_STPST_R_68  = J0030_V24_STPST_MR_68[:,0]

# J0030 NICERxXMM solutions (Vinciguerra et al. 2024)

J0030_V24_STPDT_MR_95  = np.loadtxt(f'{data_path}/J0030_Vinciguerra24_STPDT_NxXMM_95.txt')
J0030_V24_STPDT_M_95  = J0030_V24_STPDT_MR_95[:,1]
J0030_V24_STPDT_R_95  = J0030_V24_STPDT_MR_95[:,0]

J0030_V24_STPDT_MR_68 = np.loadtxt(f'{data_path}/J0030_Vinciguerra24_STPDT_NxXMM_68.txt')
J0030_V24_STPDT_M_68 = J0030_V24_STPDT_MR_68[:,1]
J0030_V24_STPDT_R_68 = J0030_V24_STPDT_MR_68[:,0]

J0030_V24_PDTU_MR_95 = np.loadtxt(f'{data_path}/J0030_Vinciguerra24_PDTU_NxXMM_95.txt')
J0030_V24_PDTU_M_95 = J0030_V24_PDTU_MR_95[:,1]
J0030_V24_PDTU_R_95 = J0030_V24_PDTU_MR_95[:,0]

J0030_V24_PDTU_MR_68 = np.loadtxt(f'{data_path}/J0030_Vinciguerra24_PDTU_NxXMM_68.txt')
J0030_V24_PDTU_M_68 = J0030_V24_PDTU_MR_68[:,1]
J0030_V24_PDTU_R_68 = J0030_V24_PDTU_MR_68[:,0]


# J0740 ST-U NICER x XMM (Riley et al. 2021)

J0740_R21_STU_MR_68 = np.loadtxt(f'{data_path}/J0740_Riley21_68.txt')
J0740_R21_STU_MR_95 = np.loadtxt(f'{data_path}/J0740_Riley21_95.txt')

J0740_R21_STU_M_68 = J0740_R21_STU_MR_68[:,1]
J0740_R21_STU_R_68 = J0740_R21_STU_MR_68[:,0]
J0740_R21_STU_M_95 = J0740_R21_STU_MR_95[:,1]
J0740_R21_STU_R_95 = J0740_R21_STU_MR_95[:,0]

# J0740 ST-U NICER x XMM (Salmi et al. 2022)


J0740_S22_STU_MR_68 = np.loadtxt(f'{data_path}/J0740_Salmi22_3C50-3X_68.txt')
J0740_S22_STU_MR_95 = np.loadtxt(f'{data_path}/J0740_Salmi22_3C50-3X_95.txt')

J0740_S22_STU_M_68 = J0740_S22_STU_MR_68[:,1]
J0740_S22_STU_R_68 = J0740_S22_STU_MR_68[:,0]
J0740_S22_STU_M_95 = J0740_S22_STU_MR_95[:,1]
J0740_S22_STU_R_95 = J0740_S22_STU_MR_95[:,0]

# J0740 ST-U NICER x XMM (Salmi et al. 2024)

J0740_S24_STU_MR_68 = np.loadtxt(f'{data_path}/J0740_Salmi24_68.txt')
J0740_S24_STU_MR_95 = np.loadtxt(f'{data_path}/J0740_Salmi24_95.txt')

J0740_S24_STU_M_68 = J0740_S24_STU_MR_68[:,1]
J0740_S24_STU_R_68 = J0740_S24_STU_MR_68[:,0]
J0740_S24_STU_M_95 = J0740_S24_STU_MR_95[:,1]
J0740_S24_STU_R_95 = J0740_S24_STU_MR_95[:,0]

# J0437 CST+PDT with AGN and background (Choudhury et al. 2024)

J0437_C24_CSTPDT_MR_68 = np.load(f'{data_path}/J0437_CST_PDT_3C50_BKG_AGN_smooth_hiMN_lowXPSI_res_3sigma_68_contours.npy')
J0437_C24_CSTPDT_MR_95 = np.load(f'{data_path}/J0437_CST_PDT_3C50_BKG_AGN_smooth_hiMN_lowXPSI_res_3sigma_95_contours.npy')

J0437_C24_CSTPDT_M_68 = J0437_C24_CSTPDT_MR_68[:,1]
J0437_C24_CSTPDT_R_68 = J0437_C24_CSTPDT_MR_68[:,0]
J0437_C24_CSTPDT_M_95 = J0437_C24_CSTPDT_MR_95[:,1]
J0437_C24_CSTPDT_R_95 = J0437_C24_CSTPDT_MR_95[:,0]

# Read in IM M-R contour data for J0030 and J0740
# Column 1 = Mass in solar mass
# Column 2 = Radius in km

IMJ0030MR_1sig = np.loadtxt(f'{data_path}/J0030_Miller19_1sigma.txt')
IMJ0030MR_2sig = np.loadtxt(f'{data_path}/J0030_Miller19_2sigma.txt')
IMJ0740MR_1sig = np.loadtxt(f'{data_path}/J0740_Miller21_1sigma.txt')
IMJ0740MR_2sig = np.loadtxt(f'{data_path}/J0740_Miller21_2sigma.txt')

IMJ0740MR_1sigM = IMJ0740MR_1sig[:,0]
IMJ0740MR_1sigR = IMJ0740MR_1sig[:,1]
IMJ0740MR_2sigM = IMJ0740MR_2sig[:,0]
IMJ0740MR_2sigR = IMJ0740MR_2sig[:,1]

IMJ0030MR_1sigM = IMJ0030MR_1sig[:,0]
IMJ0030MR_1sigR = IMJ0030MR_1sig[:,1]
IMJ0030MR_2sigM = IMJ0030MR_2sig[:,0]
IMJ0030MR_2sigR = IMJ0030MR_2sig[:,1]

### Plot M-R curves and current NICER results (XPSI only)

fig, axes = plt.subplots(1, 2, figsize=(7,3))
if all_scenarios:
    fig, axes = plt.subplots(2, 2, figsize=(7,5))
    ax_baseline = axes[0,0]
    ax_new2 = axes[1,0] # New New 2
    ax_new = axes[0,1] # New
    ax_new3 = axes[1,1]
else:
    axes = axes.reshape((1,2))
    ax_baseline = axes[0,0]
    ax_new = axes[0,1]


#Panel Baseline

#J0030 ST+PST (R19)
ax_baseline.plot(J0030_R19_STPST_R_95, J0030_R19_STPST_M_95, linewidth=1.0, color='black',  linestyle='dashed', alpha=0.4)
ax_baseline.plot(J0030_R19_STPST_R_68, J0030_R19_STPST_M_68, linewidth=1.0, color='black', linestyle='dotted', alpha=0.4)

#J0740 ST-U (R21)
ax_baseline.plot(J0740_R21_STU_R_95, J0740_R21_STU_M_95, linewidth=1.0, color='black',  linestyle='dashed', alpha=0.4)
ax_baseline.plot(J0740_R21_STU_R_68, J0740_R21_STU_M_68, linewidth=1.0, color='black', linestyle='dotted', alpha=0.4)

#J0740 ST-U (S22)
ax_baseline.fill(J0740_S22_STU_R_95, J0740_S22_STU_M_95, linewidth=1.0, color='xkcd:light violet',  alpha=0.4)
ax_baseline.fill(J0740_S22_STU_R_68, J0740_S22_STU_M_68, linewidth=1.0, color='xkcd:light violet',  alpha=0.8)

#J0030 ST+PST (V24)

ax_baseline.fill(J0030_V24_STPST_R_95, J0030_V24_STPST_M_95, linewidth=1.0, color='xkcd:aqua',  alpha=0.3)
ax_baseline.fill(J0030_V24_STPST_R_68, J0030_V24_STPST_M_68, linewidth=1.0, color='xkcd:aqua',  alpha=0.6)

# Panel New (i.e, New 2)

ax_new.fill(J0740_S24_STU_R_95, J0740_S24_STU_M_95, linewidth=1.0, color='xkcd:light violet',  alpha=0.4)
ax_new.fill(J0740_S24_STU_R_68, J0740_S24_STU_M_68, linewidth=1.0, color='xkcd:light violet',  alpha=0.8)

ax_new.fill(J0030_V24_STPDT_R_95, J0030_V24_STPDT_M_95, linewidth=1.0, color='xkcd:aqua',  alpha=0.3)
ax_new.fill(J0030_V24_STPDT_R_68, J0030_V24_STPDT_M_68, linewidth=1.0, color='xkcd:aqua',  alpha=0.6)

ax_new.fill(J0437_C24_CSTPDT_R_95, J0437_C24_CSTPDT_M_95, linewidth=1.0, color='xkcd:light red',  alpha=0.2)
ax_new.fill(J0437_C24_CSTPDT_R_68, J0437_C24_CSTPDT_M_68, linewidth=1.0, color='xkcd:light red',  alpha=0.4)

if all_scenarios:
    # Panel New 1

    ax_new2.fill(J0740_S24_STU_R_95, J0740_S24_STU_M_95, linewidth=1.0, color='xkcd:light violet',  alpha=0.4)
    ax_new2.fill(J0740_S24_STU_R_68, J0740_S24_STU_M_68, linewidth=1.0, color='xkcd:light violet',  alpha=0.8)

    ax_new2.fill(J0030_V24_STPST_R_95, J0030_V24_STPST_M_95, linewidth=1.0, color='xkcd:aqua',  alpha=0.3)
    ax_new2.fill(J0030_V24_STPST_R_68, J0030_V24_STPST_M_68, linewidth=1.0, color='xkcd:aqua',  alpha=0.6)

    ax_new2.fill(J0437_C24_CSTPDT_R_95, J0437_C24_CSTPDT_M_95, linewidth=1.0, color='xkcd:light red',  alpha=0.2)
    ax_new2.fill(J0437_C24_CSTPDT_R_68, J0437_C24_CSTPDT_M_68, linewidth=1.0, color='xkcd:light red',  alpha=0.4)

    # Panel New 3

    ax_new3.fill(J0740_S24_STU_R_95, J0740_S24_STU_M_95, linewidth=1.0, color='xkcd:light violet',  alpha=0.4)
    ax_new3.fill(J0740_S24_STU_R_68, J0740_S24_STU_M_68, linewidth=1.0, color='xkcd:light violet',  alpha=0.8)

    ax_new3.fill(J0030_V24_PDTU_R_95, J0030_V24_PDTU_M_95, linewidth=1.0, color='xkcd:aqua',  alpha=0.3)
    ax_new3.fill(J0030_V24_PDTU_R_68, J0030_V24_PDTU_M_68, linewidth=1.0, color='xkcd:aqua',  alpha=0.6)

    ax_new3.fill(J0437_C24_CSTPDT_R_95, J0437_C24_CSTPDT_M_95, linewidth=1.0, color='xkcd:light red',  alpha=0.2)
    ax_new3.fill(J0437_C24_CSTPDT_R_68, J0437_C24_CSTPDT_M_68, linewidth=1.0, color='xkcd:light red',  alpha=0.4)


# Texts 
x = 0.03
y = 0.90
ax_baseline.text(x, y, 'Baseline', transform=ax_baseline.transAxes)
ax_new.text(x, y, 'New', transform=ax_new.transAxes)
ax_baseline.annotate('J0030', xy=(9.0, 1.3), color='xkcd:aqua')
ax_baseline.annotate('J0740', xy=(9.0, 1.8), color='xkcd:light violet')
if all_scenarios:
    ax_new.annotate('J0437', xy=(9.0, 1.5), color='xkcd:light red')
    ax_new2.text(x, y, 'New 2', transform=ax_new2.transAxes)
    ax_new3.text(x, y, 'New 3', transform=ax_new3.transAxes)
else:
    ax_new.annotate('J0437', xy=(9.0, 1.5), color='xkcd:light red')

# Axis labels
xlabel = '$R$ [km]'
ylabel = r'$M$ $[M_\odot]$'
[ax.set_xlabel(xlabel) for ax in axes[-1,:]]
[ax.set_ylabel(ylabel) for ax in axes[:,0]]

# Limits
xlim = (9, 16.5)
ylim = (1.0, 2.4)
[ax.set_xlim(xlim) for ax in axes.flatten()]
[ax.set_ylim(ylim) for ax in axes.flatten()]

# Ticks
xticks = [8, 10, 12, 14, 16]
yticks = [1.0, 1.4, 1.8, 2.2]
[ax.set_xticks(xticks) for ax in axes.flatten()] # Set xticks
[ax.set_xticklabels([]) for ax in axes[:-1,:].flatten()] # Remove xticklabels from upper panels
[ax.set_yticks(yticks) for ax in axes.flatten()] # Set yticks
[ax.set_yticklabels([]) for ax in axes[:,-1]] # Remove yticklabels from right panel(s)
[ax.minorticks_on() for ax in axes.flatten()] # Set minor ticks on

# Ticks on right and top side
[plotting.right_side_ticks(ax) for ax in axes[:,1]]
[plotting.top_side_ticks(ax) for ax in axes[0,:]]

# Z-order the panels. Needed for the ticks to show up properly
# Right on top of left, top on top of bottom.
if len(axes) > 1:
    axes[1,0].set_zorder(1)
    axes[1,1].set_zorder(2)
axes[0,0].set_zorder(3)
axes[0,1].set_zorder(4)

# Save result
fig.subplots_adjust(wspace=0, hspace=0)
fname = 'MRdata_all.pdf' if all_scenarios else 'MRdata.pdf'
plotting.save(fig, f'{parent}/fig', fname, args.tmp_output)
