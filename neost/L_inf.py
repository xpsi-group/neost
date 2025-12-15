import os
import numpy as np
from neost import global_imports

from scipy.stats import norm
from scipy.optimize import curve_fit

rhons = global_imports._rhons
n_ns = global_imports._n_ns
dyncm2_to_MeVfm3 = global_imports._dyncm2_to_MeVfm3

energydensities = np.logspace(14.2, 16, 50)                # log of grams per centimeter cubed
number_density_ns = energydensities/rhons
number_density = number_density_ns * n_ns

# Load data ( for calculating the correlation, doesnt have to be done every time)
#path = os.path.dirname(__file__)    # current path
#file_path = os.path.abspath(os.path.join(path, '..', 'DANEoST/data/cEFT_band_fitting' ))
file_path = '/work/home/mm12wyxy/neost/secretneost/DANEoST/data/cEFT_band_fitting'  # only hard-coding seems to work for travelling to paths outside of where neost is actually installed

N3LO_beta = np.loadtxt(file_path+'/N3LO_beta_X2D.txt')                          ## add better path later
N3LO_pnm = np.loadtxt(file_path+'/N3LO_PNM_X2D.txt')
#N2LO_beta = np.loadtxt(file_path+'/N2LO_beta_X2D.txt')                         ## uncomment this line if working with N2LO   
#N2LO_pnm = np.loadtxt(file_path+'/N2LO_PNM_X2D.txt')

samples = 1000                                            # they are very correlated, so a smaller array would also work
quantiles = np.linspace(0.05,0.95, samples)
densities = 0.157                                         # in the beta, pnm grids is closest density to n0 available
    
def linear(x, a, b):
    return a * x + b

def correlation(beta, pnm, density, quantiles, number_density):
    n_prior = np.isclose(beta[:,0], density)
    Exp_beta = (beta[:,8] + beta[:, 9])[n_prior]  # Pressure nucleons plus electrons
    Exp_PNM  = (pnm[:,8]  + pnm[:,9])[n_prior]
    sigma_beta = beta[:,10][n_prior]              # Std
    sigma_PNM = pnm[:,10][n_prior]

    # Fit correlation
    PPF_beta = norm.ppf(q = quantiles, loc = Exp_beta, scale = sigma_beta)
    PPF_PNM  = norm.ppf(q = quantiles, loc = Exp_PNM, scale = sigma_PNM)
    popt_prior, pcov_data = curve_fit(linear, PPF_beta, PPF_PNM)

    n_posterior = np.isclose(number_density, density, atol = 0.005)   #atol somewhat random, but more precision would barely change L inferred
    return n_posterior, popt_prior, pcov_data                         #pcov_data actually never used, but good for checking precision

corr_N3LO = correlation(N3LO_beta, N3LO_pnm, densities, quantiles, number_density)   #don't forget to call N2LO if working with N2LO   

class class_L_inf:

    def __init__(self):
        print('Starting L inference')

    def corr(self):
        self.corr_N3LO = corr_N3LO
        return self.corr_N3LO

    def pnm_P(self, popt_prior, pressure, index_density):
        pressure = pressure[index_density][0,:]*dyncm2_to_MeVfm3
        return popt_prior[0] * pressure + popt_prior[1]

    def function_L_inf (self, p_corr, index_density):
        n = number_density[index_density]
        self.L = p_corr / n* 3
        return self.L
  
