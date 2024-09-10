import neost
from neost.eos import polytropes, tabulated
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from neost import PosteriorAnalysis
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib
from matplotlib import pyplot
from pymultinest.solve import solve
import time
import os
if not os.path.exists("chains"): 
   os.mkdir("chains")

import neost.global_imports as global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons

# Choose the EoS model that is used
polytropes_example = polytropes.PolytropicEoS(crust = 'ceft-Hebeler', rho_t = 1.1*rho_ns)

# Define the likelihood function
# Here just a simple 2D gaussian.

muM = 2.08   #J0740
muR = 11.155
sigM = 0.07 
sigR = 0.1 # uncertainty in radius
test = multivariate_normal(mean=[muM, muR], cov=[[sigM, 0.0], [0.0, sigR]])

likelihood_functions = [test.pdf]
likelihood_params = [['Mass', 'Radius']]

# This is not a GW event so we set chirp mass to None
chirp_mass = [None]
number_stars = len(chirp_mass)

run_name = "polytropes_test_2D_gaussian"

# We only vary the parameters 'gamma1' and 'ceft' for now
variable_params = {'gamma1':[1., 4.5], 'ceft':[polytropes_example.min_norm, polytropes_example.max_norm]}
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})

# and set the rest to static parameters in the sampling
static_params = {'gamma2':4., 'gamma3':2.6, 'rho_t1':1.8, 'rho_t2':4.}

# Then we define the prior and likelihood accordingly
prior = Prior(polytropes_example, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass)

print("Bounds of prior are")
print(variable_params)
print("number of parameters is %d" %len(variable_params))

# First we test if everything is working as expected
print("Testing prior and likelihood")
cube = np.random.rand(50, len(variable_params))
for i in range(len(cube)):
    par = prior.inverse_sample(cube[i])
    print(likelihood.call(par))
print("Testing done")

# Then we start the sampling
start = time.time()
result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=1000, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename='chains/' + run_name, verbose=True)
end = time.time()
print(end - start)

# Compute auxiliary data for posterior analysis
PosteriorAnalysis.compute_auxiliary_data('chains/' + run_name, polytropes_example, 
                                          variable_params, static_params, chirp_mass)


# Make some analysis plots
PosteriorAnalysis.cornerplot('chains/' + run_name, variable_params)
PosteriorAnalysis.mass_radius_posterior_plot('chains/' + run_name)
PosteriorAnalysis.mass_radius_prior_predictive_plot('chains/' + run_name, variable_params, label_name='+ fake J0740 data')
PosteriorAnalysis.eos_posterior_plot('chains/' + run_name, variable_params)


