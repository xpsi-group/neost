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
run_name = "PP-R24-moi-"
directory = 'runs'

pp_eos = polytropes.PolytropicEoS(crust='ceft-Keller-N3LO', rho_t=1.5*rho_ns)

# Create the likelihoods for the individual measurements
mr_J0740 = np.loadtxt(directory + '/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat').T
J0740_LL = gaussian_kde(mr_J0740)

mr_J0030 = np.loadtxt(directory + '/J0030_bravo_STPDT_NxX_lp1k_se08_mrsamples_post_equal_weights.dat').T
J0030_LL = gaussian_kde(mr_J0030)


mr_J0437 = np.loadtxt(directory + '/J0437_3C50_CST_PDT_AGN_lp20k_se03_mrsamples_post_equal_weights.dat').T
J0437_LL = gaussian_kde(mr_J0437)


# And next up is the gravitational wave event
GW170817 = np.load(directory + '/GW170817_McQL1L2weights.npy')
GW170817_LL = kalepy.KDE(GW170817[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW170817[:,4], bandwidth=0.1, kernel='gaussian')

GW190425 = np.load(directory + '/GW190425_McQL1L2weights.npy')
GW190425_LL = kalepy.KDE(GW190425[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW190425[:,4], bandwidth=0.1, kernel='gaussian')


# get kramer mass-moi data
# reshape to [mass, moi][n_samples]
# moi should be in 10^45 g cm^2 (also called I_45 in literature)
kramer_resampled = np.load(directory + '/pdf_Ip45_mp_resampled.npy')

J0737_LL = gaussian_kde(kramer_resampled) #M0 = 1.3382 (mean mass used in resample.py)
M0 = 1.3382 #used for the likelihood function!!

# Pass the likelihoods to the solver
likelihood_functions = [J0740_LL.pdf, J0030_LL.pdf,J0437_LL.pdf, J0737_LL.pdf, lambda points: GW170817_LL.density(np.array([points]).T, probability=True)[1][0],lambda points: GW190425_LL.density(np.array([points]).T, probability=True)[1][0]]
likelihood_params = [['Mass', 'Radius'], ['Mass', 'Radius'], ['Mass', 'Radius'], ['MoI']]

# Define whether event is GW or not and define number of stars/events
chirp_mass = [None,None,None,None,1.186,1.44]
number_stars = len(chirp_mass)

# Define variable parameters, same prior as previous papers of Raaijmakers et al.
variable_params={'ceft':[pp_eos.min_norm, pp_eos.max_norm],'gamma1':[0.,8],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2,8.3],'rho_t2':[2,8.3]}
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})
	
static_params = {}

prior = Prior(pp_eos, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass, M0)

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
#                n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{directory}/{run_name}', verbose=True)
# end = time.time()
# print(end - start)

compute_auxiliary_data_Thijs(directory, pp_eos, variable_params, static_params, chirp_mass, identifier=run_name, prior=False)