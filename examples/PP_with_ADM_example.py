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
run_name = "PP-example-bosonic-adm-run"

# We're exploring a polytropic (PP) EoS parametrization with a chiral effective field theory (cEFT) parametrization based on Hebeler's work
# Transition between PP parametrisation and cEFT parametrization occurs at 1.1*saturation density
polytropes_pp = polytropes.PolytropicEoS(crust = 'ceft-Hebeler', rho_t = 1.1*rho_ns, adm_type = 'Bosonic',dm_halo = False,two_fluid_tidal = False)

# Create the likelihoods for the individual measurements
# First we load the mass-radius measurement
mass_radius_j0740 = np.load('j0740.npy').T
J0740_LL = gaussian_kde(mass_radius_j0740)


# Pass the likelihoods to the solver
likelihood_functions = [J0740_LL]
likelihood_params = [['Mass', 'Radius']]

# Define whether event is GW or not and define number of stars/events
chirp_mass = [None]
number_stars = len(chirp_mass)

# Define variable parameters, same prior as previous papers of Raaijmakers et al.
if polytropes_pp.adm_type == 'Bosonic':
    variable_params={'ceft':[polytropes_pp.min_norm, polytropes_pp.max_norm],'gamma1':[1.,4.5],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[1.5,8.3],'rho_t2':[1.5,8.3],
                    'mchi':[-2, 8],'gchi_over_mphi': [-3,3],'adm_fraction':[0., 5.]}
elif polytropes_pp.adm_type == 'Fermionic':
    variable_params={'ceft':[polytropes_pp.min_norm, polytropes_pp.max_norm],'gamma1':[1.,4.5],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[1.5,8.3],'rho_t2':[1.5,8.3],
                    'mchi':[-2, 9],'gchi_over_mphi': [-5,3],'adm_fraction':[0., 5.]}

else:
    variable_params={'ceft':[polytropes_pp.min_norm, polytropes_pp.max_norm],'gamma1':[1.,4.5],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[1.5,8.3],'rho_t2':[1.5,8.3]}

#Note if the user wants to have a seperate adm_fraction per source, include it below via
#'adm_fraction_' + str(i+1):[0., 5.]. And eliminate the adm_fraction above, as that is to assume all Neutron stars have the same amount of adm_fraction
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
result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=50, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename='chains/' + run_name, verbose=True, resume=False)
end = time.time()
print(end - start)

# Compute auxiliary data for posterior analysis
PosteriorAnalysis.compute_auxiliary_data('chains/' + run_name, polytropes_pp, 
                                         variable_params, static_params, chirp_mass,dm = True)


# Make some analysis plots
PosteriorAnalysis.cornerplot('chains/' + run_name, variable_params, dm = True)
PosteriorAnalysis.mass_radius_posterior_plot('chains/' + run_name,dm = True)
PosteriorAnalysis.mass_radius_prior_predictive_plot('chains/' + run_name,variable_params, label_name='+ J0740 dataset', dm = True)
PosteriorAnalysis.eos_posterior_plot('chains/' + run_name, variable_params)
