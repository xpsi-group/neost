import neost
from neost.eos import polytropes, tabulated
from neost.Star import Star
from scipy.stats import multivariate_normal
import numpy as np
from pymultinest.solve import solve
import time
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from neost import PosteriorAnalysis_new
from neost.Prior import Prior
from neost.Likelihood import Likelihood
import neost.global_imports as global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons


eos_name = 'polytropes'
directory = 'chains'
#run_name = 'synthetic_rotation_core_measurement_run'
run_name = 'synthetic_non_rotating_measurement_run_'

# EOS = polytropes.PolytropicEoS(crust = 'ceft-Hebeler', rho_t = 2e14, adm = True)
# EOS.update({'gamma1':2.3, 'gamma2':4., 'gamma3':2.6, 'rho_t1':1.8, 'rho_t2':4.,'mchi':15000., 'gchi_over_mphi':0.1, 'adm_fraction':0., 'ceft':2.6}, max_edsc=True)

#Luuk, I have slightly modified my old script, which I commented out below in case you wanted to see it, but below I have a slightly updated version for your conversion
#The main upshot to produce your synthetic neutron star mass-radius measurements is to do the following:
# 1. pick an EOS by choosing your gammas, rhos, and ceft (I have set crust and rho_t for you since those are the most realistic nuclear physics calculations that NEoST, compared to what I used back in 2022/2023 :)), which I pick some basic examples below

EOS = polytropes.PolytropicEoS(crust = 'ceft-Keller-N3LO', rho_t = 1.5*rho_ns)
EOS.update({'gamma1':2.3, 'gamma2':4., 'gamma3':2.6, 'rho_t1':1.8, 'rho_t2':4.,'ceft':2.6}, max_edsc=True)

masses = [1.0, 1.4, 1.6, 1.8, 2.0] # these are the masses of the synthetic neutron stars you want to create, which you can modify as you see fit

# 2. Compute the mass-radius relation for your chosen EOS parameters and create an interpolator of the mass-radius relation (obviously you can do this different frequencies and such, but this is just a simple example to show you how I did it for the synthetic data in my paper, which you can modify as you see fit):
epsc = np.logspace(14.5, np.log10(EOS.max_edsc), 100)
M = np.zeros_like(epsc)
R = np.zeros_like(epsc)

freq = np.random.uniform(200, 600, len(masses))

for i, eps in enumerate(epsc):
    star = Star(eps)
    star.solve_structure(EOS.energydensities, EOS.pressures)
    M[i] = star.Mrot
    R[i] = star.Req

# 3. Create an interpolator of the mass-radius and mass-central energy density relation, which will allow you to easily create the synthetic neutron star measurements you want for a given mass. 
R_of_M = UnivariateSpline(M, R, k=1, s=0)
logEps_of_M = UnivariateSpline(M, np.log10(epsc), k=1, s=0) # this is good to have as NEoST is going to infer the central energy density of each star, so keep this in case you want to check how well it recovers the central energy density when we star incorporating your rotation corrections code. 




Radii = R_of_M(masses) # this gives you the radii of the synthetic neutron stars corresponding to the masses you chose, which you can modify as you see fit

# 4. Now you can create the synthetic neutron star measurements by assuming some uncertainty on the mass and radius measurements, which I have set to 2% in this example, but you can modify as you see fit.

muM1 = masses[0]  
muR1 = Radii[0]
sigM1 = muM1*0.02 #2% uncertainty
sigR1 = muR1*0.02  #2% uncertainty in radius
test1 = multivariate_normal(mean=[muM1, muR1], cov=[[sigM1, 0.0], [0.0, sigR1]])

muM2 = masses[1] 
muR2 = Radii[1]
sigM2 = muM2*0.02      #2% uncertainty
sigR2 = muR2*0.02     #2% uncertainty in radius
test2 = multivariate_normal(mean=[muM2, muR2], cov=[[sigM2, 0.0], [0.0, sigR2]])

muM3 = masses[2]    
muR3 = Radii[2]
sigM3 = muM3*0.02   #2% uncertainty
sigR3 = muR3*0.02  #2% uncertainty in radius 
test3 = multivariate_normal(mean=[muM3, muR3], cov=[[sigM3, 0.0], [0.0, sigR3]])

muM4 = masses[3]   
muR4 = Radii[3]
sigM4 = muM4*0.02 # 2 % uncerainty in mass 
sigR4 = muR4*0.02  #2% uncertainty in radius 
test4 = multivariate_normal(mean=[muM4, muR4], cov=[[sigM4, 0.0], [0.0, sigR4]])

muM5 = masses[4]    
muR5 = Radii[4]
sigM5 = muM5*0.02       #2% uncertainty in mass
sigR5 = muR5*0.02       #2% uncertainty in radius 
test5 = multivariate_normal(mean=[muM5, muR5], cov=[[sigM5, 0.0], [0.0, sigR5]])

likelihood_functions = [test1.pdf,test2.pdf,test3.pdf,test4.pdf,test5.pdf]
likelihood_params = [['Mass', 'Radius'],['Mass','Radius'],['Mass','Radius'],['Mass','Radius'],['Mass','Radius']]

# This is not a GW event so we set chirp mass to None
chirp_mass = [None,None,None,None,None]
number_stars = len(chirp_mass)





variable_params={'gamma1':[0.,8.],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2.,8.3],'rho_t2':[2.,8.3],'ceft':[EOS.min_norm, EOS.max_norm]}
for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})

# Define static parameters, empty dict because all params are variable 
static_params={}

prior = Prior(EOS, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass, freq)

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
# result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=20, evidence_tolerance=0.1,
#                n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{directory}/{run_name}', verbose=True, resume=False)
# end = time.time()
# print(end - start)

# Compute auxiliary data for posterior analysis
PosteriorAnalysis_new.compute_auxiliary_data(directory, EOS, variable_params, static_params, chirp_mass, identifier=run_name)

# Make some analysis plots
PosteriorAnalysis_new.cornerplot(directory, variable_params, identifier=run_name)
PosteriorAnalysis_new.mass_radius_posterior_plot(directory, identifier=run_name)
PosteriorAnalysis_new.mass_radius_prior_predictive_plot(directory, variable_params, identifier=run_name, label_name='+ J0740 dataset')

PosteriorAnalysis_new.eos_posterior_plot(directory, variable_params, identifier=run_name)
plt.show()