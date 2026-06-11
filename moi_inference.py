import neost
import kalepy
from neost.eos import polytropes, tabulated
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from neost.PosteriorAnalysis_Thijs import compute_auxiliary_data_Thijs
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
import numpy as np
from pymultinest.solve import solve
import time
import os

import neost.global_imports as global_imports

# Some physical constants
c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons

# Define name for run, extra - at the end is for nicer formatting of output
run_name = "PP-moi-only"
directory = 'runs'

pp_eos = polytropes.PolytropicEoS(crust='ceft-Keller-N3LO', rho_t=1.5*rho_ns)

# get kramer mass-moi data
# reshape to [mass, moi][n_samples]
# moi should be in 10^45 g cm^2 (also called I_45 in literature)
kramer_resampled = np.load('data/pdf_Ip45_mp_resampled.npy')

J0737_LL = gaussian_kde(kramer_resampled).pdf

likelihood_functions = [J0737_LL]
likelihood_params = [['MoI']]

chirp_mass = [None]
n_stars = len(chirp_mass)

# Define variable parameters, same prior as Rutherford et al.
variable_params={'ceft':[pp_eos.min_norm, pp_eos.max_norm],'gamma1':[0,8],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2,8.3],'rho_t2':[2,8.3]}
for i in range(n_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})
	
static_params = {}

prior = Prior(pp_eos, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass, M0=1.3382)

print("Bounds of prior are")
print(variable_params)
print("number of parameters is %d" %len(variable_params))

# Perform a test, this will draw 50 random points from the prior and calculate their likelihood
print("Testing prior and likelihood")
cube = np.random.rand(50, len(variable_params))
for i in range(len(cube)):
    par = prior.inverse_sample(cube[i])
    print(likelihood.call(par))
print("Testing done")

# # Then we start the sampling with MultiNest
# start = time.time()
# result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=5000, evidence_tolerance=0.1,
#                n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{directory}/{run_name}', verbose=True, resume=False)
# end = time.time()
# print(end - start)

compute_auxiliary_data_Thijs(directory, pp_eos, variable_params, static_params, chirp_mass, identifier=run_name, prior=False)