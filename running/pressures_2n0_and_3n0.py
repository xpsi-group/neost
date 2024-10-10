# Standard libraries
import argparse

# 3rd party
import matplotlib
import numpy as np

from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import norm

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--order', required=True, type=str)
parser.add_argument('-pp', '--piecewise_polytropes', action='store_true')
parser.add_argument('-cs', '--speedofsound', action='store_true')
parser.add_argument('-s', '--core_start', required=True, type=float)
parser.add_argument('-c', '--case', required=True, type=str)
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

# Store arguments
case = args.case.lower()
pp = args.piecewise_polytropes
cs = args.speedofsound
order = args.order.lower()
core_start = args.core_start
core_start_int = int(core_start*10)

# Check argument sanity
if pp and cs:
    raise ValueError('Choose either PP or CS!')
elif not pp and not cs:
    raise ValueError('Choose either PP or CS!')
if 'n2lo' != order and 'n3lo' != order and 'hebeler' != order and 'nlo-0.5' != order and 'nlo-0.7' != order:
    raise ValueError(f'Unknown chiral order {order}')
if case != 'prior' and case != 'baseline' and case != 'new' and case != 'new2' and case != 'new3':
    raise ValueError(f'Case {case} currently not implemented')

# Set variables
parametrization = None
if pp:
    parametrization = 'pp'
elif cs:
    parametrization = 'cs'

# Input/output directory
directory = 'results' if not args.repro else 'repro'
if case == 'prior':
    directory = f'{directory}/prior/{parametrization}/{order}/{core_start_int}'
else:
    directory = f'{directory}/posterior/{parametrization}/{order}/{core_start_int}/{case}'

# Interpolation
rho_ns = 267994004080000.03 # cgs, saturation mass density

def eos_interpolator(pressures):
    num_eos = len(pressures[0])
    pressure_2n0 = np.zeros(len(pressures[0]))
    pressure_3n0 = np.zeros((len(pressures[0])))
    for i in range(len(pressures[0])):
        densities = np.logspace(14.2,16,50) / rho_ns # in units of rho_ns (effectively a ratio w.r.t saturation mass density)
        nonzero_indices = np.nonzero(pressures[:,i])[0]
        p = pressures[:,i]
        p = p[nonzero_indices] # only grab non-zero values
        densities = densities[nonzero_indices]
        try:
            p_dens =UnivariateSpline(densities, p, k=1, s=0, ext=1) # requires x to be strictly increasing, but a much faster function than interp1d, so we prefer to use it
            pressure_2n0[i] = p_dens(2.)
            pressure_3n0[i] = p_dens(3.)
        except ValueError:
            raise Warning("densities is not strictly increasing! Switching to slower interp1d() function")
            p_dens = interp1d(densities, p, kind='linear', fill_value= [0.],bounds_error = False) #back-up interpolator in the event x is not strictly increasing (Safety net)
            pressure_2n0[i] = p_dens(2.)                                                        
            pressure_3n0[i] = p_dens(3.)

    pressure_2n0 = pressure_2n0[pressure_2n0>0.]
    pressure_3n0 = pressure_3n0[pressure_3n0>0.]
    return pressure_2n0,pressure_3n0

pressures = np.load(f'{directory}/pressures.npy')
pressn20, pressn30 = eos_interpolator(pressures)

fname = f'{directory}/press_n20.npy'
print(f'Saving pressures at 2 n_0 to {fname}')
np.save(fname, pressn20)

fname = f'{directory}/press_n30.npy'
print(f'Saving pressures at 3 n_0 to {fname}')
np.save(fname, pressn30)
