import neost
from neost.eos import polytropes
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from neost import PosteriorAnalysis
from scipy.stats import multivariate_normal, gaussian_kde
import numpy as np
from pymultinest.solve import solve
import time
import os
import pathlib


import neost.global_imports as global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons



EOS = polytropes.PolytropicEoS(crust = 'ceft-Keller-N3LO', rho_t = 1.1*rho_ns, adm_type = 'Dark Energy')


# Create the likelihoods for the individual measurements
mass_radius_j0740 = np.load('j0740.npy').T
J0740_LL = gaussian_kde(mass_radius_j0740)


likelihood_functions = [J0740_LL.pdf]
likelihood_params = [['Mass', 'Radius']]

# This is not a GW event so we set chirp mass to None
chirp_mass = [None]
number_stars = len(chirp_mass)

run_name = "Dark_energy_posterior_example_"
directory = f'{run_name}/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist



#lower_bound_rho_plus = 1.5*rho_ns --> right down to the N3LO chiral EFT band
#upper_bound_rho_plus = 10**(16)/rho_ns = 37.31426766180507 #taken to the an energy density that captures all of the maximum central energy densities for the entire PP parameterization
variable_params = {'gamma1':[1.,4.5],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[1.5,8.3],'rho_t2':[1.5,8.3],
                  'A_param':[0.1, 0.7],'rho_plus': [1.1,37.3142677],'alpha':[0.1, 1.],'ceft':[EOS.min_norm, EOS.max_norm]}


for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})




static_params = {}
# In[ ]:


prior = Prior(EOS, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass)

print("Bounds of prior are")
print(variable_params)
print("with model "+eos_name)
print("number of parameters is %d" %len(variable_params))

## TESTING ##
print("Testing prior and likelihood")
cube = np.random.rand(500, len(variable_params))
for i in range(len(cube)):
    par = prior.inverse_sample(cube[i])
    print(likelihood.call(par),i)
print("Testing done")


# In[ ]:


start = time.time()
result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=3000, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{run_name}/{run_name}', verbose=True)
end = time.time()
print(end - start)

print('Solving done')



PosteriorAnalysis.compute_auxiliary_data(directory, EOS, variable_params, static_params, chirp_mass, dm=False, de=True, sampler='multinest', identifier=run_name)

PosteriorAnalysis.compute_table_data(directory, EOS, variable_params, static_params, dm=False, de=True, sampler='multinest', identifier=run_name)

def get_quantiles(array, quantiles=[0.025, 0.5, 0.975]):
        contours = np.nanquantile(array, quantiles) #changed to nanquantile to inorder to ignore the nans that may appear
        low = contours[0]
        median = contours[1]
        high = contours[2]
        minus = low - median
        plus = high - median
        return np.round(median,2),np.round(plus,2),np.round(minus,2) 

Data_array = np.loadtxt(run_name + 'table_data.txt')
print('M_TOV: ', get_quantiles(Data_array[:,0]))
print('R_TOV: ', get_quantiles(Data_array[:,1]))
print('R_1.4: ', get_quantiles(Data_array[:,2]))
print('R_2.0: ', get_quantiles(Data_array[:,3]))
print('Delta R = R_2.0 - R_1.4: ', get_quantiles(Data_array[:,3] - Data_array[:,2]))
