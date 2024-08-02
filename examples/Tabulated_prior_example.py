# # Tabulated Prior Example
# The machinary and the explanation of them is identical to those found in the Piecewise Polytropic, Speed of Sound, and Tabulated Examples.

import neost
from neost.eos import tabulated
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

# Define name for run
run_name = "prior-tabulated-"

# We're exploring a tabulated (T) EoS 
eos_name = 'tabulated'

baryondensity, pressure_B, energydensity_B = np.loadtxt('ap4_new.dat', unpack=True) #in units of g/cm^3 for all values

pressure_B = pressure_B*c**2 #(in units of g/(cm s^2))

energydensity_B = energydensity_B

# Defining the base EoS model
tabulated_example = tabulated.TabulatedEoS(energydensity_B,pressure_B)
tabulated_example.update({}, max_edsc=True)

# Create the likelihoods for the individual measurements
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

#Only vary the central densities on the sources because there are no parameters to vary in a tabulated equation of state
variable_params = {}
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})

# Define static parameters, empty dict because all params are variable 
static_params={}

# Define prior
prior = Prior(tabulated_example, variable_params, static_params, chirp_mass)

print("Bounds of prior are")
print(variable_params)
print("number of parameters is %d" %len(variable_params))

# Define likelihood, pseudo_var is required as input because NEoST expects to be able to pass the parameter sample drawn from the prior to be passable to the likelihood
def loglike(pseudo_var):
    return 1.

# Then we start the sampling, note the greatly increased number of livepoints, this is required because each livepoint terminates after 1 iteration
start = time.time()
result = solve(LogLikelihood=loglike, Prior=prior.inverse_sample, n_live_points=100000, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename='chains/' + run_name, verbose=True, resume=False)
end = time.time()
print(end - start)

# Compute auxiliary data for posterior analysis
PosteriorAnalysis.compute_auxiliary_data('chains/' + run_name, tabulated_example,
                                         variable_params, static_params, chirp_mass)

# Make some analysis plots
PosteriorAnalysis.cornerplot('chains/' + run_name, variable_params)
PosteriorAnalysis.mass_radius_posterior_plot('chains/' + run_name)

#Will throw an error stating the the underling equation of state is fixed because there are no parameters to vary
#for a tabulated EoS
#PosteriorAnalysis.mass_radius_prior_predictive_plot('chains/' + run_name,variable_params, label_name='+ Synthetic dataset')
#PosteriorAnalysis.eos_posterior_plot('chains/' + run_name,variable_params)

