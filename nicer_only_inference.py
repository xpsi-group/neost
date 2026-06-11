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
run_name = "NICER_ONLY_"
directory = 'runs'

pp_eos = polytropes.PolytropicEoS(crust='ceft-Keller-N3LO', rho_t=1.5*rho_ns)

# Create the likelihoods for the individual measurements
mr_J0740 = np.loadtxt(directory + '/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat').T
J0740_LL = gaussian_kde(mr_J0740)

mr_J0030 = np.loadtxt(directory + '/J0030_bravo_STPDT_NxX_lp1k_se08_mrsamples_post_equal_weights.dat').T
J0030_LL = gaussian_kde(mr_J0030)

mr_J0437 = np.loadtxt(directory + '/J0437_3C50_CST_PDT_AGN_lp20k_se03_mrsamples_post_equal_weights.dat').T
J0437_LL = gaussian_kde(mr_J0437)

likelihood_functions = [J0740_LL.pdf, J0030_LL.pdf,J0437_LL.pdf]
likelihood_params = [['Mass', 'Radius'], ['Mass', 'Radius'], ['Mass', 'Radius']]

chirp_mass = [None, None, None]
number_stars = len(chirp_mass)

variable_params={'gamma1':[0.,8],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2,8.3],'rho_t2':[2,8.3], 'ceft':[pp_eos.min_norm, pp_eos.max_norm]}
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})
	
static_params = {}

prior = Prior(pp_eos, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass, M0=1.4)

print("Bounds of prior are")
print(variable_params)
print("number of parameters is %d" %len(variable_params))

# # Perform a test, this will draw 50 random points from the prior and calculate their likelihood
# print("Testing prior and likelihood")
# cube = np.random.rand(50, len(variable_params))
# for i in range(len(cube)):
#     par = prior.inverse_sample(cube[i])
#     print(likelihood.call(par))
# print("Testing done")

compute_auxiliary_data_Thijs(directory, pp_eos, variable_params, static_params, chirp_mass, identifier=run_name, prior=False)