import neost
import kalepy
from neost.eos import polytropes, tabulated
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from neost import PosteriorAnalysis
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
run_name = "PP-example-run-"
directory = 'chains'

# We're exploring a polytropic (PP) EoS parametrization with a chiral effective field theory (cEFT) parametrization based on Hebeler's work
# Transition between PP parametrisation and cEFT parametrization occurs at 1.1*saturation density
polytropes_pp = polytropes.PolytropicEoS(crust = 'ceft-Hebeler', rho_t = 1.1*rho_ns)

# Create the likelihoods for the individual measurements
# First we load the mass-radius measurement
mass_radius_j0740 = np.load('j0740.npy').T
J0740_LL = gaussian_kde(mass_radius_j0740)
# And next up is the gravitational wave event
GW170817 = np.load('GW170817.npy')
GW170817_LL = kalepy.KDE(GW170817[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW170817[:,4], bandwidth=0.1, kernel='gaussian', diagonal=True)

# Pass the likelihoods to the solver
likelihood_functions = [J0740_LL, lambda points: GW170817_LL.density(np.array([points]).T, probability=True)[1][0]]
likelihood_params = [['Mass', 'Radius']]

# Define whether event is GW or not and define number of stars/events
chirp_mass = [None, 1.186]
number_stars = len(chirp_mass)

# Define variable parameters, same prior as previous papers of Raaijmakers et al.
variable_params={'ceft':[polytropes_pp.min_norm, polytropes_pp.max_norm],'gamma1':[1.,4.5],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[1.5,8.3],'rho_t2':[1.5,8.3]}
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})

# Define static parameters, empty dict because all params are variable 
static_params={}

# Define joint prior and joint likelihood
prior = Prior(polytropes_pp, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass)

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

# Then we start the sampling with MultiNest
start = time.time()
result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=5000, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{directory}/{run_name}', verbose=True, resume=False)
end = time.time()
print(end - start)

# Compute auxiliary data for posterior analysis
PosteriorAnalysis.compute_auxiliary_data(directory, polytropes_pp, variable_params, static_params, chirp_mass, identifier=run_name)


# Make some analysis plots
PosteriorAnalysis.cornerplot(directory, variable_params, identifier=run_name)
PosteriorAnalysis.mass_radius_posterior_plot(directory, identifier=run_name)
PosteriorAnalysis.mass_radius_prior_predictive_plot(directory, variable_params, identifier=run_name, label_name='+ J0740 dataset')
PosteriorAnalysis.eos_posterior_plot(directory, variable_params, identifier=run_name)
