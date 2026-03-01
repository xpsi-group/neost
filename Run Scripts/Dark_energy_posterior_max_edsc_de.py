import neost
from neost.eos import polytropes
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from neost import PosteriorAnalysis
from scipy.stats import multivariate_normal, gaussian_kde
import numpy as np
import matplotlib
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot
from pymultinest.solve import solve
import time
import os
import pathlib
import corner as corner


# In[ ]:


import neost.global_imports as global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons



EOS = polytropes.PolytropicEoS(crust = 'ceft-Keller-N3LO', rho_t = 1.5*rho_ns, adm_type = 'Dark Energy')


mr_J0740 = np.load('Riley00740_mr_posterior_samples.npy').T
J0740_LL = gaussian_kde(mr_J0740)

mr_J0030 = np.load('Riley0030_mr_posterior_samples.npy').T
J0030_LL = gaussian_kde(mr_J0030)


likelihood_functions = [J0740_LL.pdf, J0030_LL.pdf]
likelihood_params = [['Mass', 'Radius'],['Mass','Radius']]

# This is not a GW event so we set chirp mass to None
chirp_mass = [None,None]
number_stars = len(chirp_mass)

run_name = "Dark_energy_posterior_max_edsc_de"
directory = f'{run_name}/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist


variable_params = {'gamma1':[0., 8.], 'gamma2':[0., 8.], 'gamma3':[0.5, 8.], 'rho_t1':[1.5, 8.3], 'rho_t2':[1.5, 8.3],'A_param':[0.1, 0.7],'rho_plus': [1.5,37.3142677],'alpha':[0.1, 1.],'ceft':[EOS.min_norm, EOS.max_norm]}


for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})



static_params = {}
# In[ ]:


prior = Prior(EOS, variable_params, static_params, chirp_mass, dark_energy = True)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass)

eos_name = 'polytropes'

print("Bounds of prior are")
print(variable_params)
print("with model "+eos_name)
print("number of parameters is %d" %len(variable_params))

## TESTING ##
print("Testing prior and likelihood")
cube = np.random.rand(15, len(variable_params))
for i in range(len(cube)):
    par = prior.inverse_sample(cube[i])
    print(likelihood.call(par),i)
print("Testing done")


# In[ ]:


start = time.time()
result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=1000, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{run_name}/{run_name}', verbose=True)
end = time.time()
print(end - start)

print('Solving done')

PosteriorAnalysis.cornerplot(directory, variable_params, dm=False, sampler='multinest', identifier=run_name)

# PosteriorAnalysis.compute_auxiliary_data_de(run_name, EOS,
#                                          variable_params, static_params, prior = False)

# PosteriorAnalysis.compute_table_data_de(run_name, EOS, variable_params, static_params)

# def get_quantiles(array, quantiles=[0.025, 0.5, 0.975]):
#         contours = np.nanquantile(array, quantiles) #changed to nanquantile to inorder to ignore the nans that may appear
#         low = contours[0]
#         median = contours[1]
#         high = contours[2]
#         minus = low - median
#         plus = high - median
#         return np.round(median,2),np.round(plus,2),np.round(minus,2) 

# Data_array = np.loadtxt(run_name + 'table_data.txt')
# print('M_TOV: ', get_quantiles(Data_array[:,0]))
# print('R_TOV: ', get_quantiles(Data_array[:,1]))
# print('R_1.4: ', get_quantiles(Data_array[:,2]))
# print('R_2.0: ', get_quantiles(Data_array[:,3]))
# print('Delta R = R_2.0 - R_1.4: ', get_quantiles(Data_array[:,3] - Data_array[:,2]))
