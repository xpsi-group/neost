import neost
import kalepy
from neost.eos import speedofsound, tabulated
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from neost import PosteriorAnalysis
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib
from matplotlib import pyplot
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
run_name = "CS-example-run-"

# We're exploring a speed of sound (CS) EoS parametrization with a chiral effective field theory (cEFT) parametrization based on Hebeler's work
# Transition between CS parametrisation and cEFT parametrization occurs at 1.1*saturation density
speedofsound_cs = speedofsound.SpeedofSoundEoS(crust = 'ceft-Hebeler', rho_t = 1.1*rho_ns)

# Create the likelihoods for the individual measurements
# First we load the mass-radius measurement
mass_radius_j0740 = np.load('j0740.npy').T
J0740_LL = gaussian_kde(mass_radius_j0740)
# And next up is the gravitational wave event
GW170817 = np.load('GW170817.npy')
GW170817_LL = kalepy.KDE(GW170817[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW170817[:,4], bandwidth=0.1, kernel='gaussian')

# Pass the likelihoods to the solver
likelihood_functions = [J0740_LL, lambda points: GW170817_LL.density(np.array([points]).T, probability=True)[1][0]]
likelihood_params = [['Mass', 'Radius']]

# Define whether event is GW or not and define number of stars/events
chirp_mass = [None, 1.186]
number_stars = len(chirp_mass)

# Define variable parameters, same prior as previous papers of Raaijmakers et al.
variable_params={'ceft':[speedofsound_cs.min_norm, speedofsound_cs.max_norm],'a1':[0.1,1.5],'a2':[1.5,12.],'a3/a2':[0.05,2.],'a4':[1.5,37.],'a5':[0.1,1.]}
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})

# Define static parameters, empty dict because all params are variable 
static_params={}

# Define joint prior and joint likelihood
prior = Prior(speedofsound_cs, variable_params, static_params, chirp_mass)
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
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename='chains/' + run_name, verbose=True, resume=False)
end = time.time()
print(end - start)

# Compute auxiliary data for posterior analysis
PosteriorAnalysis.compute_auxiliary_data('chains/' + run_name, speedofsound_cs, 
                                         variable_params, static_params, chirp_mass)


# Make some analysis plots
PosteriorAnalysis.cornerplot('chains/' + run_name, variable_params)
PosteriorAnalysis.mass_radius_posterior_plot('chains/' + run_name)
PosteriorAnalysis.mass_radius_prior_predictive_plot('chains/' + run_name,variable_params, label_name='+ J0740 dataset')
PosteriorAnalysis.eos_posterior_plot('chains/' + run_name,variable_params)
