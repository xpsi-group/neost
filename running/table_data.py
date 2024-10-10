# Standard libraries
import argparse

# 3rd party
import numpy as np

# Local
import likelihood_definitions
import variable_parameters
from neost.eos import polytropes, speedofsound
from neost import PosteriorAnalysis
import neost.global_imports as global_imports

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--order', required=True, type=str)             ####### CAUTION: make modifications in PosteriorAnalysis.py before running for NLO!
parser.add_argument('-pp', '--piecewise_polytropes', action='store_true')
parser.add_argument('-cs', '--speedofsound', action='store_true')
parser.add_argument('-s', '--core_start', required=True, type=float)
parser.add_argument('-c', '--case', required=True, type=str)
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

case = args.case.lower()
pp = args.piecewise_polytropes
cs = args.speedofsound
order = args.order.upper()
core_start = args.core_start
core_start_int = int(core_start*10)
parametrization = None
if pp:
    parametrization = 'PP'
elif cs:
    parametrization = 'CS'

# Some physical constants
rho_ns = global_imports._rhons

# Set values
rho_t = core_start * rho_ns
crust = 'ceft-Hebeler' if 'hebeler' == order.lower() else f'ceft-Keller-{order}'

# Input/output directory
directory = 'results' if not args.repro else 'repro'
if case == 'prior':
    directory = f'{directory}/prior/{parametrization}/{order}/{core_start_int}/'.lower()
else:
    directory = f'{directory}/posterior/{parametrization}/{order}/{core_start_int}/{case}/'.lower()
print(f'Computing {directory}/table_data.txt based on {directory}/post_equal_weights.dat')

# Define EOS parametrization
eos = None
if pp:
    eos = polytropes.PolytropicEoS(crust=crust, rho_t=rho_t)
elif cs:
    eos = speedofsound.SpeedofSoundEoS(crust=crust, rho_t=rho_t)

# Define individual likelihoods - we only need number of stars here
data_path = 'data/likelihood_data'
_, _, chirp_mass = likelihood_definitions.get_likelihood(case, data_path)
number_stars = len(chirp_mass)

# Define variable parameters
variable_params = variable_parameters.get_variable_params(parametrization, eos, core_start, number_stars)

# Define static parameters, empty dict because all params are variable 
static_params={}

# Compute the data
PosteriorAnalysis.compute_table_data(f'{directory}', eos, variable_params, static_params)
