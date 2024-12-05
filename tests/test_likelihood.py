import pytest
import neost
from neost.eos import polytropes, tabulated
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
import neost.global_imports as global_imports
from scipy.stats import multivariate_normal
import numpy as np

class TestLikelihoodCheck(object):
    def test_likelihood_is_correct(self): 
        c = global_imports._c
        G = global_imports._G
        Msun = global_imports._M_s
        pi = global_imports._pi
        rho_ns = global_imports._rhons
        polytropes_example = polytropes.PolytropicEoS(crust = 'ceft-Hebeler', rho_t = 1.1*rho_ns)
        muM = 2.08
        muR = 11.155
        sigM = 0.07 
        sigR = 0.1
        test = multivariate_normal(mean=[muM, muR], cov=[[sigM, 0.0], [0.0, sigR]])
        likelihood_functions = [test.pdf]
        likelihood_params = [['Mass', 'Radius']]
        chirp_mass = [None]
        number_stars = len(chirp_mass)
        variable_params = {'gamma1':[1., 4.5], 'ceft':[polytropes_example.min_norm, polytropes_example.max_norm]}
        for i in range(number_stars):
	        variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})
        static_params = {'gamma2':4., 'gamma3':2.6, 'rho_t1':1.8, 'rho_t2':4.}
        prior = Prior(polytropes_example, variable_params, static_params, chirp_mass)
        likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass)
        cube = [0.5,0.5,0.5]
        par = prior.inverse_sample(cube)
        ll = likelihood.call(par)
        print(ll)
        if np.isclose(-4.187612429216115, ll, rtol=1.0e-5) is False:
            print('C TOV solvers either not installed or broken, using Python TOV solver instead')
            assert np.isclose(--4.192614304543445, ll, rtol=1.0e-5)
            
        else:
            assert np.isclose(-4.187612429216115, ll, rtol=1.0e-5)

