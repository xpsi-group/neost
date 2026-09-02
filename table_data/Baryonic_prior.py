import neost
from neost.eos import polytropes
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from neost import PosteriorAnalysis
from scipy.stats import multivariate_normal, gaussian_kde
import numpy as np
from pymultinest.solve import solve
import time
import os
import pathlib

from pathlib import Path

import neost.global_imports as global_imports

# Some physical constants
c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons


script_dir = Path(__file__).resolve().parent

eos_name = 'polytropes'

EOS = polytropes.PolytropicEoS(crust = 'ceft-Keller-N3LO', rho_t = 1.5*rho_ns)

likelihood_functions = []
likelihood_params = [['Mass', 'Radius']]

# Define whether event is GW or not and define number of stars/events
chirp_mass = [None]
number_stars = len(chirp_mass)


run_name = "Baryonic_prior_"
repro_path = os.getcwd()
# repro_path = script_dir.parent / f'{run_name}/'
# repro_path.mkdir(parents=True, exist_ok=True).mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

print(f"Folder created at: {repro_path}")



variable_params = {'ceft':[EOS.min_norm, EOS.max_norm], 'gamma1':[0.,8.],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2.,8.3],'rho_t2':[2.,8.3]}

for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})


static_params={}

# Define prior
prior = Prior(EOS, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params,chirp_mass)

print("Bounds of prior are")
print(variable_params)
print("number of parameters is %d" %len(variable_params))


# start = time.time()
# result = solve(LogLikelihood=likelihood.loglike_prior, Prior=prior.inverse_sample, n_live_points=100000, evidence_tolerance=0.1,
#                n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{repro_path}/{run_name}', verbose=True)
# end = time.time()
# print(end - start)

print('Testing done')
print('Moving on to posterior analysis')


PosteriorAnalysis.compute_table_data(repro_path, EOS, variable_params, static_params, dm=False, de=False, sampler='multinest', identifier=run_name)
# PosteriorAnalysis.compute_auxiliary_data(repro_path, EOS, variable_params, static_params, chirp_mass, dm=False, de=False, sampler='multinest', identifier=run_name)
