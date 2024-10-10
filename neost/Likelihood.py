import numpy
from scipy.interpolate import UnivariateSpline

from neost.Star import Star
from neost import global_imports
from neost.utils import m1_from_mc_m2

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons
n_ns = global_imports._n_ns


class Likelihood():

    def __init__(self, prior, likelihood_functions,
                 likelihood_params, chirp_masses):

        self.prior = prior
        self.likelihood_functions = likelihood_functions
        self.likelihood_params = likelihood_params
        self.chirp_masses = chirp_masses

    def call(self, pr):

        likelihoods = []
        pr_dict = self.prior.pr

        constraints = self.prior.EOS.check_constraints()

        if constraints is False:
            return -1e101

        for i in range(self.prior.number_stars):
            star = Star(10**(pr_dict['rhoc_' + str(i + 1)]), 0.0)
            star.solve_structure(self.prior.EOS.energydensities,
                                 self.prior.EOS.pressures)

            if (star.Mrot > 3. or star.Mrot < 1. or star.Req > 16. or star.Req < star.Mrot * Msun * 2. * G / c**2. / 1e5): 
            #if (star.Mrot > 3. or star.Req > 16. or star.Req < star.Mrot * Msun * 2. * G / c**2. / 1e5): 
                return -1e101

            #if (self.max_M < 2.):                                                  ######### to mimic Ingo's distribution
            #    print(self.max_M)
            #    return -1e101

            if self.chirp_masses[i] is None:
                MassRadiusTidal = {'Mass':star.Mrot, 'Radius':star.Req,
                                   'Lambda':star.tidal}
                tmp = list(map(MassRadiusTidal.get, self.likelihood_params[i]))
                like = self.likelihood_functions[i](tmp)
                like = self.array_to_scalar(like)
                likelihoods.append(like)
            else:
                M2 = star.Mrot
                M1 = m1_from_mc_m2(self.chirp_masses[i], M2)
                if M1 < M2 or M1 > self.prior.MRT[:,0][-1]:
                    return -1e101
                MTspline = UnivariateSpline(self.prior.MRT[:,0],
                                            self.prior.MRT[:,2], k=1, s=0)                 #with NLO, there's not enough points here to make this interpolation?
                point = numpy.array([self.chirp_masses[i], M2 / M1,
                                     MTspline(M1), star.tidal])
                like = self.likelihood_functions[i](point)
                like = self.array_to_scalar(like)
                likelihoods.append(like)

        like_total = numpy.prod(numpy.array(likelihoods))
        if like_total == 0.0:
            return -1e101

        return numpy.log(like_total)
    
    def array_to_scalar(self, var):
        # numpy no longer accepts ragged arrays,
        # check that if var is an array it only has one element and return it
        if hasattr(var, '__len__'):
            try:
                assert(len(var) == 1)
            except AssertionError:
                raise ValueError('Malformed likelihood')
            var = var[0]
        return var

    def loglike_prior(self,pr):
        pr_dict = self.prior.pr
        constraints = self.prior.EOS.check_constraints()
        if constraints is False:
            return -1e101

        star = Star(self.prior.EOS.max_edsc)
        star.solve_structure(self.prior.EOS.energydensities,
                                 self.prior.EOS.pressures)
        #if(star.Mrot < 2):                                                ##### more similar framework to Ingo's?
        #    return -1e101

        for i in range(self.prior.number_stars):
            star = Star(10**(pr_dict['rhoc_' + str(i + 1)]), 0.0)
            star.solve_structure(self.prior.EOS.energydensities,
                                 self.prior.EOS.pressures)
            if(star.Mrot < 1.):
                return -1e101
            
        loglike_const = 1.
        return loglike_const


    def loglike_prior_gaussian(self,pr):
        pr_dict = self.prior.pr
        #print(pr_dict)
        #print(pr_dict['ceft'])
        
        loglikelihood=[]                           #they're not all the same, we need to calculate them individually then multiply them
        
        constraints = self.prior.EOS.check_constraints()
        if constraints is False:
            return -1e101

        star = Star(self.prior.EOS.max_edsc)
        star.solve_structure(self.prior.EOS.energydensities,
                                 self.prior.EOS.pressures)
        if(star.Mrot < 1):                        #checks if the max mass is smaller than Msun
                return -1e101

        for i in range(self.prior.number_stars):
            star = Star(10**(pr_dict['rhoc_' + str(i + 1)]), 0.0)
            star.solve_structure(self.prior.EOS.energydensities,
                                 self.prior.EOS.pressures)
            if(star.Mrot < 1.):                   #checks if the sampled central density returns mass smaller than Msun
                return -1e101

        for i in pr_dict:                         #the likelihood is not constant, but should have a gaussian shape
            if i != 'ceft':
                loglike_const = loglikelihood.append(1.)  
            else:
                loglike_gaus = loglikelihood.append(numpy.exp(-(pr_dict[i]-2.6315)**2/(2*0.4245)))  #hard-coded to n3lo now so that it recovers previous ceft fit (centered around k_average, sigma^2=k_max-k_average, gamma fixed)  
        like_total = numpy.prod(numpy.array(loglikelihood))
        #print(like_total)
        if like_total == 0.0:
            return -1e101

        return numpy.log(like_total)
        #we can run just this function and return the individual likelihoods before the product (as sanity check)


    def loglike_prior_gaussian_v2(self,pr):            #a gaussian distribution of likelihoods both for norm and index in the ceft band
        pr_dict = self.prior.pr
       	loglikelihood=[]                               #they're not all the same, we need to calculate them individually then multiply them
        aux_loglikelihood_ceft=[]
        aux_loglikelihood_ceft_ind=[]

       	constraints = self.prior.EOS.check_constraints()
        if constraints is False:
            return -1e101

        star = Star(self.prior.EOS.max_edsc)
        star.solve_structure(self.prior.EOS.energydensities,
                                 self.prior.EOS.pressures)
        if(star.Mrot < 1):                             #checks if the max mass is smaller than Msun
                return -1e101

        for i in range(self.prior.number_stars):
            star = Star(10**(pr_dict['rhoc_' + str(i + 1)]), 0.0)
            star.solve_structure(self.prior.EOS.energydensities,
                                 self.prior.EOS.pressures)
            if(star.Mrot < 1.):                        #checks if the sampled central density returns mass smaller than Msun
                return -1e101

        for i in pr_dict:                              #the likelihood is not constant, but should have a gaussian shape
            if i != 'ceft' or 'ceft_in':               #must add a new parameter to the sampling
                loglike_const = loglikelihood.append(1.)
            if i =='ceft':
                aux=numpy.exp(-(pr_dict[i]-2.6315)**2/(2*0.4245))    #hard-coded to n3lo now so that it recovers previous ceft fit (centered around k_average, sigma^2=k_max-k_average)
                aux_loglikelihood_ceft.append(aux)        #only this value, for sanity check later
                loglike_gaus = loglikelihood.append(aux)

            if i =='ceft_in':

                index = ((pr_dict['ceft'] - 2.207) / (3.056 - 2.207) * (2.814 - 2.361) + 2.361)  #are the parameters sampled in order? #hard-coded to n3lo values
                aux = numpy.exp(-(index)**2/(2*0.2265))     #(sigma^2=index_max-index_av, may be too large)
                
                if aux < 2.361:    #for each sampled norm, index is gaussian around the value it used to be, unless it goes over the range, in this case, manually set to min/max limits so we don't lose the fitting
                    aux=2.361
                if aux > 2.814:
                    aux=2.814

                loglike_gaus_in = loglikelihood.append(aux)
                aux_loglikelihood_ceft_in.append(aux) 

        like_total = numpy.prod(numpy.array(loglikelihood))
        #print(like_total)
        if like_total == 0.0:
            return -1e101

        return numpy.log(like_total)    #later must also return aux_loglikelihood_ceft and aux_loglikelihood_ceft_in
