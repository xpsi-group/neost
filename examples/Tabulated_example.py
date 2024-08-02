import neost
from neost.eos import tabulated
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from neost import PosteriorAnalysis
import numpy as np
import random
import matplotlib
from matplotlib import pyplot
from pymultinest.solve import solve
import time
import os

import neost.global_imports as global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons



eos_name = 'tabulated'

baryondensity, pressure_B, energydensity_B = np.loadtxt('ap4_new.dat', unpack=True) #in units of g/cm^3

pressure_B = pressure_B*c**2 #(in units of g/(cm s^2))

energydensity_B = energydensity_B

# Defining the base EoS model
tabulated_example = tabulated.TabulatedEoS(energydensity_B,pressure_B)
tabulated_example.update({}, max_edsc=True)

# Here just a simple 2D gaussian.

muM = 2.0947   #synthetic source with log_10(central energy density) = 15.240180657118929
muR = 10.808
sigM = 0.05*muM # 5% uncertainty in mass 
sigR = 0.05*muR # 5% uncertainty in radius
test = multivariate_normal(mean=[muM, muR], cov=[[sigM, 0.0], [0.0, sigR]])

muM2 = 1.7090   #synthetic source with log_10(central energy density) = 15.07681219601247
muR2 = 11.312
sigM2 = 0.05*muM2 # 5% uncertainty in mass 
sigR2 = 0.05*muR2 # 5% uncertainty in radius
test2 = multivariate_normal(mean=[muM2, muR2], cov=[[sigM2, 0.0], [0.0, sigR2]])

muM3 = 1.0814   #synthetic source with log_10(central energy density) = 14.913443734906012
muR3 = 11.4587
sigM3 = 0.05*muM3 # 5% uncertainty in mass 
sigR3 = 0.05*muR3 # 5% uncertainty in radius
test3 = multivariate_normal(mean=[muM3, muR3], cov=[[sigM3, 0.0], [0.0, sigR3]])

likelihood_functions = [test.pdf,test2.pdf,test3.pdf]
likelihood_params = [['Mass', 'Radius'],['Mass', 'Radius'],['Mass', 'Radius']]

# This is not a GW event so we set chirp mass to None
chirp_mass = [None, None, None]
number_stars = len(chirp_mass)

run_name = "tabulated_AP4_test_2D_gaussian"

# We only vary the parameters 'gamma1' and 'ceft' for now
variable_params = {}
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})

# and set the rest to static parameters in the sampling
static_params = {}

# Then we define the prior and likelihood accordingly
prior = Prior(tabulated_example, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass)

print("Bounds of prior are")
print(variable_params)
print("number of parameters is %d" %len(variable_params))

# First we test if everything is working as expected
print("Testing prior and likelihood")
cube = np.random.rand(50, len(variable_params))
for i in range(len(cube)):
    par = prior.inverse_sample(cube[i])
print("Testing done")

# Then we start the sampling
start = time.time()
result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=1000, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename='chains/' + run_name, verbose=True, resume=False)
end = time.time()
print(end - start)

# Compute auxiliary data for posterior analysis
PosteriorAnalysis.compute_auxiliary_data('chains/' + run_name, tabulated_example, 
                                          variable_params, static_params, chirp_mass)


# Make some analysis plots
PosteriorAnalysis.cornerplot('chains/' + run_name, variable_params)
PosteriorAnalysis.mass_radius_posterior_plot('chains/' + run_name)

#Cannot make the other Posterior Analysis files since EoS is tabulated and cannot be varied
#PosteriorAnalysis.mass_radius_prior_predictive_plot('chains/' + run_name,variable_params, label_name='+ Synethetic MR data')
#PosteriorAnalysis.eos_posterior_plot('chains/' + run_name,variable_params)
