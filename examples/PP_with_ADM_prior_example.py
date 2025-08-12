import neost
from neost.eos import polytropes
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
if not os.path.exists("chains"): 
   os.mkdir("chains")

import neost.global_imports as global_imports

# Some physical constants
c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons

# Define name for run
run_name = "prior-hebeler-pp-bosonic-adm-"
directory = 'chains'

# We're exploring a polytropic (P) EoS parametrization with a chiral effective field theory (CEFT) parametrization based on Hebeler's work
# Transition between CS parametrisation and CEFT parametrization occurs at 1.1*saturation density
polytropes_pp = polytropes.PolytropicEoS(crust = 'ceft-Hebeler', rho_t = 2e14, adm_type = 'Bosonic',dm_halo = False,two_fluid_tidal = False)

# Create the likelihoods for the individual measurements
mass_radius_j0740 = np.load('j0740.npy').T
J0740_LL = gaussian_kde(mass_radius_j0740)

# Pass the likelihoods to the solver
likelihood_functions = [J0740_LL.pdf]
likelihood_params = [['Mass', 'Radius']]

# Define whether event is GW or not and define number of stars/events
chirp_mass = [None]
number_stars = len(chirp_mass)

# Define variable parameters, same prior as previous papers of Raaijmakers et al

print(polytropes_pp.adm_type)
if polytropes_pp.adm_type == 'Bosonic':
    variable_params={'ceft':[polytropes_pp.min_norm, polytropes_pp.max_norm],'gamma1':[1.,4.5],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[1.5,8.3],'rho_t2':[1.5,8.3],
                    'mchi':[-2, 8],'gchi_over_mphi': [-3,3],'adm_fraction':[0., 5.]}
elif polytropes_pp.adm_type == 'Fermionic':
    variable_params={'ceft':[polytropes_pp.min_norm, polytropes_pp.max_norm],'gamma1':[1.,4.5],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[1.5,8.3],'rho_t2':[1.5,8.3],
                    'mchi':[-2, 9],'gchi_over_mphi': [-5,3],'adm_fraction':[0., 1.7]}

if polytropes_pp.adm_type == 'None':
    variable_params={'ceft':[polytropes_pp.min_norm, polytropes_pp.max_norm],'gamma1':[1.,4.5],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[1.5,8.3],'rho_t2':[1.5,8.3]}

#Note if the user wants to have a seperate adm_fraction per source, include it below via
#'adm_fraction_' + str(i+1):[0., 5.]. And eliminate the adm_fraction above, as that is to assume all Neutron stars have the same amount of adm_fraction
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})

# Define static parameters, empty dict because all params are variable 
static_params={}

# Define prior
prior = Prior(polytropes_pp, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass)

print("Bounds of prior are")
print(variable_params)
print("number of parameters is %d" %len(variable_params))

# Then we start the sampling, note the greatly increased number of livepoints, this is required because each livepoint terminates after 1 iteration
start = time.time()
result = solve(LogLikelihood=likelihood.loglike_prior, Prior=prior.inverse_sample, n_live_points=15000, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{directory}/{run_name}', verbose=True, resume=False)
end = time.time()
print(end - start)

# Compute auxiliary data for posterior analysis
PosteriorAnalysis.compute_auxiliary_data(directory, polytropes_pp, variable_params, static_params, chirp_mass, dm=True, identifier=run_name)
PosteriorAnalysis.cornerplot(directory, variable_params, dm=True, identifier=run_name)
PosteriorAnalysis.eos_posterior_plot(directory, variable_params, identifier=run_name)
