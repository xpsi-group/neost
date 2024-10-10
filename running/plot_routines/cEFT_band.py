#FYI this routine fits p(\rho=mass density), not p(n=number density)

# Standard libraries
import argparse
import os

# 3rd party
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.optimize as spopt

# Local
import plotting

# Paths
current_path = os.path.dirname(__file__)
parent = os.path.dirname(current_path)

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tmp_output', action='store_true')
args = parser.parse_args()

rho_ns = 267994004080000.03 #g/cm^3
m_n = 1.67492749804e-24 #neutron bare mass (g)

BPS = np.array([[2.97500000e-14, 6.30300000e-24, 4.40700000e-12], [3.06900000e-14, 6.30300000e-23, 4.54700000e-12], [4.36900000e-14, 7.55100000e-22, 6.47200000e-12],
                           [6.18700000e-14, 8.73700000e-21, 9.15000000e-12], [1.70000000e-13, 1.06100000e-19, 2.51600000e-11], [7.93700000e-13, 3.63200000e-18, 1.18300000e-10],
                           [4.33100000e-12, 1.18600000e-16, 6.41600000e-10], [3.93400000e-11, 6.08100000e-15, 5.82500000e-09], [9.88100000e-11, 3.10000000e-14, 1.46300000e-08],
                           [2.48300000e-10, 1.51700000e-13, 3.67500000e-08], [6.23500000e-10, 7.18300000e-13, 9.22800000e-08], [1.56600000e-09, 3.28600000e-12, 2.31900000e-07],
                           [3.93400000e-09, 1.44700000e-11, 5.82500000e-07], [9.88100000e-09, 6.08800000e-11, 1.46300000e-06], [2.48300000e-08, 2.44100000e-10, 3.67600000e-06],
                           [3.12500000e-08, 3.28200000e-10, 4.62700000e-06], [6.23500000e-08, 8.95500000e-10, 9.23300000e-06], [1.24400000e-07, 2.39200000e-09, 1.84200000e-05],
                           [2.48200000e-07, 6.27800000e-09, 3.67600000e-05], [4.95300000e-07, 1.62500000e-08, 7.33700000e-05], [9.88100000e-07, 4.16600000e-08, 1.46400000e-04],
                           [1.24400000e-06, 5.45300000e-08, 1.84300000e-04], [1.97200000e-06, 1.01700000e-07, 2.92200000e-04], [3.12500000e-06, 1.89000000e-07, 4.63100000e-04],
                           [3.93400000e-06, 2.57700000e-07, 5.83000000e-04], [4.95200000e-06, 3.14300000e-07, 7.34200000e-04], [6.23500000e-06, 4.28100000e-07, 9.24500000e-04],
                           [9.88100000e-06, 7.93800000e-07, 1.46500000e-03], [1.56600000e-05, 1.47000000e-06, 2.32300000e-03], [2.48200000e-05, 2.72200000e-06, 3.68300000e-03],
                           [3.12500000e-05, 3.53300000e-06, 4.63700000e-03], [3.93400000e-05, 4.80700000e-06, 5.83600000e-03], [4.95200000e-05, 6.54000000e-06, 7.35300000e-03],
                           [6.23500000e-05, 8.89300000e-06, 9.25600000e-03], [7.85000000e-05, 1.20900000e-05, 1.16600000e-02], [9.88100000e-05, 1.56200000e-05, 1.46800000e-02],
                           [1.24400000e-04, 2.12400000e-05, 1.84800000e-02], [1.56600000e-04, 2.88800000e-05, 2.32800000e-02], [1.97200000e-04, 3.71300000e-05, 2.93100000e-02],
                           [2.48200000e-04, 5.04800000e-05, 3.69200000e-02], [3.12500000e-04, 6.86500000e-05, 4.64900000e-02], [3.93400000e-04, 9.33000000e-05, 5.85200000e-02],
                           [4.95300000e-04, 1.26900000e-04, 7.37600000e-02], [6.23500000e-04, 1.62100000e-04, 9.28400000e-02], [6.90600000e-04, 1.80500000e-04, 1.02900000e-01],
                           [7.85000000e-04, 2.05300000e-04, 1.16900000e-01], [9.88100000e-04, 2.79100000e-04, 1.47300000e-01], [1.24400000e-03, 3.63000000e-04, 1.85500000e-01],
                           [1.60800000e-03, 4.87100000e-04, 2.39800000e-01], [1.95400000e-03, 5.21200000e-04, 2.91700000e-01], [2.46900000e-03, 5.67800000e-04, 3.68800000e-01],
                           [2.97400000e-03, 6.13500000e-04, 4.44300000e-01], [3.63300000e-03, 6.75900000e-04, 5.42700000e-01], [4.46400000e-03, 7.60100000e-04, 6.67300000e-01],
                           [5.49100000e-03, 8.73100000e-04, 8.20700000e-01], [2.50000000e-02, 5.79900000e-03, 3.76200000e+00], [5.00000000e-02, 1.16600000e-02, 7.53000000e+00],
                           [7.50000000e-02, 2.08500000e-02, 1.13000000e+01], [1.00000000e-01, 3.21600000e-02, 1.50800000e+01], [1.25000000e-01, 4.51500000e-02, 1.88600000e+01],
                           [1.50000000e-01, 5.96100000e-02, 2.26400000e+01], [1.75000000e-01, 7.54400000e-02, 2.64200000e+01], [2.00000000e-01, 9.26000000e-02, 3.02100000e+01],
                           [2.25000000e-01, 1.11000000e-01, 3.40000000e+01], [2.50000000e-01, 1.30800000e-01, 3.77900000e+01], [2.75000000e-01, 1.51800000e-01, 4.15800000e+01],
                           [3.00000000e-01, 1.74200000e-01, 4.53800000e+01], [3.25000000e-01, 1.98000000e-01, 4.91800000e+01], [3.50000000e-01, 2.23100000e-01, 5.29800000e+01],
                           [3.75000000e-01, 2.49700000e-01, 5.67800000e+01], [4.00000000e-01, 2.77700000e-01, 6.05800000e+01], [4.25000000e-01, 3.07300000e-01, 6.43800000e+01],
                           [4.50000000e-01, 3.38500000e-01, 6.81900000e+01], [4.75000000e-01, 3.71300000e-01, 7.20000000e+01], [5.00000000e-01, 4.05400000e-01, 7.58100000e+01],
                           [5.79200000e-01, 4.34030555e-01, 8.78844243e+01], [6.37066667e-01, 5.50694991e-01, 9.67134788e+01]])
BPS_pressure = BPS[:,1]
BPS_density = BPS[:,0]
BPS_energy = BPS[:,2]

def polytrope(x, n, gamma):
    return n*(x**gamma)

#for setting bounds on number density, but still fit p(\rho)
def updating_bands(band, name, bounds):
    aux={}

    aux["density_original"]= band[:,0]
    aux.update({"p_lower_original": band[:,1]})
    aux.update({"p_upper_original": band[:,2]})
    aux.update({"rho_original": ((band[:,0]*0.16e+39)*m_n)/rho_ns}) #using \rho=n*m_n

    #now bound rho and its corresponding pressure
    ########### lower
    band2 = band[band[:,0]>=bounds[0]]  #number density larger than bounds[0]
    band2 = band2[band2[:,0]<=bounds[1]] #number density smaller than bounds[1]

    #proper quantities considering the bounds
    aux.update({"density": band2[:,0]})
    aux.update({"p_lower": band2[:,1]})
    ########### upper
    aux.update({"p_upper": band2[:,2]})
    aux.update({"rho": ((band2[:,0]*0.16e+39)*m_n)/rho_ns})
    return (aux, band2)

def band_fitting_rho(band, name, bounds):

    aux=updating_bands(band, name, bounds)
    # perform fits
    result_P_lower = spopt.curve_fit(polytrope, aux[0]["rho"], aux[0]["p_lower"])
    P_lower_fit = polytrope(aux[0]["rho"], *result_P_lower[0])
    # P upper
    result_P_upper = spopt.curve_fit(polytrope, aux[0]["rho"], aux[0]["p_upper"])
    P_upper_fit = polytrope(aux[0]["rho"], *result_P_upper[0])
    return (result_P_upper, result_P_lower)


data_path = f'{parent}/data/cEFT_band_fitting'

muon_electron_N2LO = np.loadtxt(f'{data_path}/muon_electron_N2LO.txt')
muon_electron_N2LO[:,0] /= 0.16

New_N3LO = np.loadtxt(f'{data_path}/New_N3LO.txt')

# fitting
N2LO_upper, N2LO_lower = band_fitting_rho(muon_electron_N2LO, 'N2LO mu, e', [0.5,1.5])
N3LO_upper, N3LO_lower = band_fitting_rho(New_N3LO, 'New N3LO mu, e', [0.5,1.5])

# Plotting
aux_rho = np.linspace(0.5792, 1.1, 300)

#Hebeler
min_n = 1.676
max_n = 2.814
min_gamma = 2.486
max_gamma = 2.571

Hebeler_P_lower_fit = polytrope(aux_rho, min_n, min_gamma)
Hebeler_P_upper_fit = polytrope(aux_rho, max_n, max_gamma)

#the bounded data and fitting
aux_N3LO=updating_bands(New_N3LO, 'New N3LO mu, e', [0.5,1.5])
aux_N2LO=updating_bands(muon_electron_N2LO, 'N2LO mu, e', [0.5,1.5])

N3LO_P_lower_fit = polytrope(aux_N3LO[0]["rho"], *N3LO_lower[0])
N3LO_P_upper_fit = polytrope(aux_N3LO[0]["rho"], *N3LO_upper[0])

N2LO_P_lower_fit = polytrope(aux_N2LO[0]["rho"], *N2LO_lower[0])
N2LO_P_upper_fit = polytrope(aux_N2LO[0]["rho"], *N2LO_upper[0])

c_n2lo = plotting.c_n2lo
c_hebeler = plotting.c_hebeler
c_n3lo_11 = plotting.c_n3lo_11

fig = plt.figure(figsize=(3.6,2.8))
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')

# BPS
ax.plot(BPS_density[:-2], BPS_pressure[:-2], color='black', label='BPS')

# Make sure that BPS is lower than the chiral EFT band
assert(BPS_pressure[-3] < Hebeler_P_lower_fit[0])

# Interpolation
interpolation_color = 'xkcd:shocking pink'
rhotrans = [0.5, 0.5088, 0.5176, 0.5264, 0.5352, 0.544, 0.5528, 0.5616, 0.5704, 0.5792]
prestrans_lower = [0.4054, 0.40837726, 0.41132477, 0.41424333, 0.4171337, 0.4199966, 0.42283274, 0.42564278, 0.42842738, 0.43118714]
prestrans_upper = [0.4054, 0.43188957, 0.45961088, 0.4885988, 0.51888857, 0.5505158, 0.58351648, 0.61792696, 0.65378394, 0.69112452]
ax.plot(rhotrans, prestrans_lower, color=interpolation_color, ls='dotted', lw=1.0)
ax.plot(rhotrans, prestrans_upper, color=interpolation_color, ls='dotted', lw=1.0)

# Chiral EFT bands
ax.fill_between(aux_N2LO[0]['density'][aux_N2LO[0]['density']>0.5792], N2LO_P_lower_fit[aux_N2LO[0]['density']>0.5792], N2LO_P_upper_fit[aux_N2LO[0]['density']>0.5792], color=c_n2lo[0], alpha=1.0, label='N$^2$LO')
ax.fill_between(aux_N3LO[0]['density'][aux_N3LO[0]['density']>0.5792], N3LO_P_lower_fit[aux_N3LO[0]['density']>0.5792], N3LO_P_upper_fit[aux_N3LO[0]['density']>0.5792], color=c_n3lo_11[0], alpha=1.0, label='N$^3$LO')
ax.fill_between(aux_rho, Hebeler_P_lower_fit, Hebeler_P_upper_fit, color='grey', alpha=0.5)
lw = 1.0
ax.plot(aux_rho, Hebeler_P_lower_fit, ls='dotted', color='black', lw=lw)
ax.plot(aux_rho, Hebeler_P_upper_fit, ls='dotted', color='black', lw=lw)

# Legend
interpolation = plotting.custom_line(interpolation_color, linestyle='dotted', lw=1.0)
bps_line = plotting.custom_line('black')
hebeler_patch = mpatches.Patch(facecolor='#bfbfbf', edgecolor='black', linestyle='dotted', lw=1.0, fill=True)
n2lo_patch = plotting.custom_patch(c_n2lo[0])
n3lo_patch = plotting.custom_patch(c_n3lo_11[0])
custom_lines = [bps_line, interpolation, hebeler_patch, n2lo_patch, n3lo_patch]
labels = ['BPS', r'BPS to $\chi$EFT band', 'Hebeler et al.', 'N$^2$LO', 'N$^3$LO', ]

ax.legend(custom_lines, labels, loc='upper left', frameon=False)

ax.set_xlabel(r'$n \, [n_{\mathrm{0}}]$')
ax.set_ylabel(r'$P \, [\mathrm{MeV}/\mathrm{fm}^3]$')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

ax.set_xlim(0.2, 1.5)
ax.set_ylim(5e-2, 1.9e1)

x = np.linspace(0.6,0.68)
y = 2 * np.ones_like(x)

ax.axvline(0.5792, color='xkcd:gray', lw=0.5)
ax.axvline(1.1, color='xkcd:gray', lw=0.5)

ax.minorticks_on()
ax.set_xticks([0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 1.3, 1.4],minor=True)
ax.set_xticks([0.2, 0.6, 1.1, 1.5])
ax.set_xticklabels([0.2, 0.6, 1.1, 1.5])
ax.set_xticklabels([], minor=True)

ax2 = ax.secondary_yaxis('right')
ax2.tick_params(axis='y', which='both', direction='in')
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels([])
ax2.set_zorder(10)

ax3 = ax.secondary_xaxis('top')
ax3.tick_params(axis='x', which='both', direction='in')
ax3.set_xticks(ax.get_xticks())
ax3.set_xticks(ax.get_xticks(minor=True), minor=True)
ax3.set_xticklabels([])
ax3.set_xticklabels([], minor=True)
ax3.set_zorder(10)

# Save result
plotting.save(fig, f'{parent}/fig', 'P_vs_n.pdf', args.tmp_output)
