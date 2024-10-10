# Standard libraries
import argparse
import os
import pathlib
import sys
import time

# 3rd party
import numpy as np
from pymultinest.solve import solve

# Local
import neost                               ########### CAUTION: important alterations on base.py to make NLO bands run, currently not implemented
from neost.eos import speedofsound
from neost.eos import polytropes
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood    ########### CAUTION: modified to M_TOV >= 2 M_sun
from neost import PosteriorAnalysis        ########### CAUTION: alterations to NLO needed here too, maybe
import neost.global_imports as global_imports
import likelihood_definitions
import variable_parameters
                                                                ################ For compute_cs, need to specify directory and file name (as run_name) by hand
def print_info(parametrization, order, rho_t, n_live_points, crust, directory, run_name, core_start, variable_params, case):
    print(f'parametrization = {parametrization}')
    print(f'order = {order}')
    print(f'core_start = {core_start}')
    print(f'rho_t = {rho_t}')
    print(f'n_live_points = {n_live_points}')
    print(f'crust = {crust}')
    print(f'output directory = {directory}')
    print(f'run_name = {run_name}')
    print('Bounds of prior are:')
    print(variable_params)
    print(f'Number of parameters is {len(variable_params)}')
    print(f'Run case is {case}')

# Some physical constants
c = global_imports._c
rho_ns = global_imports._rhons

# Path stuff
current_path = os.path.dirname(__file__)
data_path = 'data/likelihood_data'

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--order', required=True, type=str)
parser.add_argument('-pp', '--piecewise_polytropes', action='store_true')
parser.add_argument('-cs', '--speedofsound', action='store_true')
parser.add_argument('-s', '--core_start', required=True, type=float)
parser.add_argument('-n', '--n_live_points', type=int)
parser.add_argument('-d', '--directory', type=str)
parser.add_argument('-name', '--name', type=str)
#parser.add_argument('-id', '--id', type=str)
parser.add_argument('-c', '--case', required=True, type=str)
args = parser.parse_args()

# Store arguments
case = args.case.lower()
pp = args.piecewise_polytropes
cs = args.speedofsound
order = args.order
order_lower = order.lower()
order_nice = order.upper() if order_lower != 'hebeler' else 'Hebeler'
core_start = args.core_start
core_start_int = int(core_start*10)
n_default = 100000 if case == 'prior' else 5000
n_live_points = n_default if args.n_live_points is None else args.n_live_points

# Check argument sanity
if pp and cs:
    raise ValueError('Choose either PP or CS!')
elif not pp and not cs:
    raise ValueError('Choose either PP or CS!')
if 'n2lo' != order_lower and 'n3lo' != order_lower and 'hebeler' != order_lower and 'nlo' != order_lower and 'lo' != order_lower:
    raise ValueError(f'Unknown chiral order {order}')
if case != 'prior' and case != 'baseline' and case != 'new' and case != 'new2' and case != 'new3':
    raise ValueError(f'Case {case} currently not implemented')

# Set variables
parametrization = None
if pp:
    parametrization = 'PP'
elif cs:
    parametrization = 'CS'
rho_t = core_start * rho_ns
crust = 'ceft-Hebeler' if 'hebeler' == order_lower else f'ceft-Keller-{order_nice}'
run_name = '' if args.name is None else args.name # Use empty run name by default

# Output directory
directory = args.directory
if directory is None:
    # No directory specified, use default
    base = 'prior' if case == 'prior' else 'posterior'
    directory = f'repro/{base}/{parametrization}/{order}/{core_start_int}'.lower()
    if base == 'posterior':
        directory = f'{directory}/{case}'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

# Define EOS
eos = None
if pp:
    eos = polytropes.PolytropicEoS(crust=crust, rho_t=rho_t)
elif cs:
    eos = speedofsound.SpeedofSoundEoS(crust=crust, rho_t=rho_t)

# Define individual likelihoods
likelihood_functions, likelihood_params, chirp_mass = likelihood_definitions.get_likelihood(case, data_path)
number_stars = len(chirp_mass)

# Define variable parameters
variable_params = variable_parameters.get_variable_params(parametrization, eos, core_start, number_stars)

# Define static parameters, empty dict because all params are variable
static_params={}

# Define joint prior and joint likelihood
prior = Prior(eos, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass)

# Print run info
print_info(parametrization, order, rho_t, n_live_points, crust, directory, run_name, core_start, variable_params, case)

# Start sampling
func = likelihood.loglike_prior if case == 'prior' else likelihood.call
start = time.time()
result = solve(LogLikelihood=func, Prior=prior.inverse_sample, n_live_points=n_live_points, evidence_tolerance=0.1, n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{directory}/{run_name}', resume=True, verbose=True)  ##verbose=True is not making any difference
end = time.time()
print(end - start)

# Compute auxiliary data for posterior analysis
start = time.time()
if case == 'prior':
    #PosteriorAnalysis.compute_cs(f'{directory}/{run_name}', eos, variable_params, static_params)
    PosteriorAnalysis.compute_prior_auxiliary_data(f'{directory}/{run_name}', eos, variable_params, static_params)
    #PosteriorAnalysis.compute_table_data(f'{directory}/{run_name}', eos, variable_params, static_params)
else:
    PosteriorAnalysis.compute_auxiliary_data(f'{directory}/{run_name}', eos, variable_params, static_params, chirp_mass)
end = time.time()
print(end-start)

PosteriorAnalysis.compute_table_data(f'{directory}/{run_name}', eos, variable_params, static_params)

