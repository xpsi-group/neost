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
import random
import timeit

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons


eos_name = 'polytropes'
directory = 'chains'
run_name = 'synthetic_rotation_core_measurement_run'
#run_name = 'synthetic_non_rotating_measurement_run_'
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
M_cor = np.zeros((len(epsc), len(masses)))
R_cor = np.zeros((len(epsc), len(masses))) 


freq = np.random.uniform(200, 600, len(masses))


#freq=[500,500,500,500,500]

for i, eps in enumerate(epsc):

    star = Star(eps)
    star.solve_structure(EOS.energydensities, EOS.pressures)
    M[i] = star.Mrot
    R[i] = star.Req
    for j in range(len(masses)):
        star=Star(eps)
        star.solve_structure(EOS.energydensities, EOS.pressures, rot_cor='Uni_rel', freq=freq[j])

        M_cor[i, j] = star.Mrot
        R_cor[i, j] = star.Req
    

R_cor_of_M1= interp1d(M_cor[:,0], R_cor[:,0], kind='linear')
R_cor_of_M2= interp1d(M_cor[:,1], R_cor[:,1], kind='linear')
R_cor_of_M3= interp1d(M_cor[:,2], R_cor[:,2], kind='linear')
R_cor_of_M4= interp1d(M_cor[:,3], R_cor[:,3], kind='linear')
R_cor_of_M5= interp1d(M_cor[:,4], R_cor[:,4], kind='linear')

logEps_of_M = UnivariateSpline(M, np.log10(epsc), k=1, s=0) # this is good to have as NEoST is going to infer the central energy density of each star, so keep this in case you want to check how well it recovers the central energy density when we star incorporating your rotation corrections code. 




muM1_cor = masses[0]  
muR1_cor = R_cor_of_M1(masses[0])
sigM1_cor = muM1_cor*0.02 #2% uncertainty
sigR1_cor = muR1_cor*0.02  #2% uncertainty in radius
test1_cor = multivariate_normal(mean=[muM1_cor, muR1_cor], cov=[[sigM1_cor, 0.0], [0.0, sigR1_cor]])

muM2_cor = masses[1]  
muR2_cor = R_cor_of_M2(masses[1])
sigM2_cor = muM2_cor*0.02 #2% uncertainty
sigR2_cor = muR2_cor*0.02  #2% uncertainty in radius
test2_cor = multivariate_normal(mean=[muM2_cor, muR2_cor], cov=[[sigM2_cor, 0.0], [0.0, sigR2_cor]])

muM3_cor = masses[2]  
muR3_cor = R_cor_of_M3(masses[2])
sigM3_cor = muM3_cor*0.02 #2% uncertainty
sigR3_cor = muR3_cor*0.02  #2% uncertainty in radius
test3_cor = multivariate_normal(mean=[muM3_cor, muR3_cor], cov=[[sigM3_cor, 0.0], [0.0, sigR3_cor]])

muM4_cor = masses[3]  
muR4_cor = R_cor_of_M4(masses[3])
sigM4_cor = muM4_cor*0.02 #2% uncertainty
sigR4_cor = muR4_cor*0.02  #2% uncertainty in radius
test4_cor = multivariate_normal(mean=[muM4_cor, muR4_cor], cov=[[sigM4_cor, 0.0], [0.0, sigR4_cor]])

muM5_cor = masses[4]  
muR5_cor = R_cor_of_M5(masses[4])
sigM5_cor = muM5_cor*0.02 #2% uncertainty
sigR5_cor = muR5_cor*0.02  #2% uncertainty in radius
test5_cor = multivariate_normal(mean=[muM5_cor, muR5_cor], cov=[[sigM5_cor, 0.0], [0.0, sigR5_cor]])

likelihood_functions = [test1_cor.pdf,test2_cor.pdf,test3_cor.pdf,test4_cor.pdf,test5_cor.pdf]
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



# t1 = timeit.timeit(lambda: np.random.uniform(200, 600, len(masses)), number=1000000)
# t2 = timeit.timeit(lambda: [np.random.random() * 400 + 200 for _ in masses], number=1000000)

# print(f"NumPy:    {t1:.3f}s")
# print(f"For loop: {t2:.3f}s")

# Then we start the sampling with MultiNest
start = time.time()
result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=50, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{directory}/{run_name}', verbose=True, resume=False)
end = time.time()
print(end - start)




# Compute auxiliary data for posterior analysis
PosteriorAnalysis_new.compute_auxiliary_data(directory, EOS, variable_params, static_params, chirp_mass, identifier=run_name, freq=freq)


# Make some analysis plots
PosteriorAnalysis_new.cornerplot(directory, variable_params, identifier=run_name)
PosteriorAnalysis_new.mass_radius_posterior_plot(directory, identifier=run_name)
PosteriorAnalysis_new.mass_radius_prior_predictive_plot(directory, variable_params, identifier=run_name, label_name='+ J0740 dataset')

PosteriorAnalysis_new.eos_posterior_plot(directory, variable_params, identifier=run_name)
plt.show()