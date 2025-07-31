import neost
from neost.eos import speedofsound
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
run_name = "prior-hebeler-cs-"
directory = 'chains'

# We're exploring a speed of sound (CS) EoS parametrization with a chiral effective field theory (CEFT) parametrization based on Hebeler's work
# Transition between CS parametrisation and CEFT parametrization occurs at 1.1*saturation density
speedofsound_cs = speedofsound.SpeedofSoundEoS(crust = 'ceft-Hebeler', rho_t = 1.1*rho_ns)

# Create the likelihoods for the individual measurements
mass_radius_j0740 = np.load('j0740.npy').T
J0740_LL = gaussian_kde(mass_radius_j0740)

# Pass the likelihoods to the solver
likelihood_functions = [J0740_LL.pdf]
likelihood_params = [['Mass', 'Radius']]

# Define whether event is GW or not and define number of stars/events
chirp_mass = [None, None]
number_stars = len(chirp_mass)

# Define variable parameters, same prior as previous papers of Raaijmakers et al
variable_params={'ceft':[speedofsound_cs.min_norm, speedofsound_cs.max_norm],'a1':[0.1,1.5],'a2':[1.5,12.],'a3/a2':[0.05,2.],'a4':[1.5,37.],'a5':[0.1,1.]}
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})

# Define static parameters, empty dict because all params are variable 
static_params={}

# Define prior
prior = Prior(speedofsound_cs, variable_params, static_params, chirp_mass)

print("Bounds of prior are")
print(variable_params)
print("number of parameters is %d" %len(variable_params))

# Define likelihood, pseudo_var is required as input because NEoST expects to be able to pass the parameter sample drawn from the prior to be passable to the likelihood
def loglike(pseudo_var):
    return 1.

# No prior & likelihood test, there is no likelihood after all
# print("Testing prior and likelihood")
# cube = np.random.rand(50, len(variable_params))
# for i in range(len(cube)):
#     par = prior.inverse_sample(cube[i])
#     print(likelihood.call(par))
# print("Testing done")

# Then we start the sampling, note the greatly increased number of livepoints, this is required because each livepoint terminates after 1 iteration
start = time.time()
result = solve(LogLikelihood=loglike, Prior=prior.inverse_sample, n_live_points=100000, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{directory}/{run_name}', verbose=True, resume=False)
end = time.time()
print(end - start)

# Compute auxiliary data for posterior analysis
PosteriorAnalysis.compute_auxiliary_data(directory, speedofsound_cs,
                                         variable_params, static_params, chirp_mass, identifier=run_name)


# Make some analysis plots
PosteriorAnalysis.cornerplot(directory, variable_params, identifier=run_name)
PosteriorAnalysis.mass_radius_posterior_plot(directory, identifier=run_name)
PosteriorAnalysis.mass_radius_prior_predictive_plot(directory, variable_params, identifier=run_name label_name='+ J0740 dataset')
PosteriorAnalysis.eos_posterior_plot(directory, variable_params, identifier=run_name)
