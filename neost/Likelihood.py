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
                return -1e101


            if self.chirp_masses[i] is None:
                MassRadiusTidal = {'Mass':star.Mrot, 'Radius':star.Req,
                                   'Lambda':star.tidal}
                tmp = list(map(MassRadiusTidal.get, self.likelihood_params[i]))
                like = self.likelihood_functions[i](tmp)
                likelihoods.append(like)
            else:
                M2 = star.Mrot
                M1 = m1_from_mc_m2(self.chirp_masses[i], M2)
                if M1 < M2 or M1 > self.prior.MRT[:,0][-1]:
                    return -1e101
                MTspline = UnivariateSpline(self.prior.MRT[:,0],
                                            self.prior.MRT[:,2], k=1, s=0)
                point = numpy.array([self.chirp_masses[i], M2 / M1,
                                     MTspline(M1), star.tidal])
                like = self.likelihood_functions[i](point)
                likelihoods.append(like)

        like_total = numpy.prod(numpy.array(likelihoods))
        # print('lnlike is', like_total)
        if like_total == 0.0:
            return -1e101

        return numpy.log(like_total)
    

    def loglike_prior(self,pr):
        pr_dict = self.prior.pr
        constraints = self.prior.EOS.check_constraints()
        if constraints is False:
            return -1e101

        star = Star(self.prior.EOS.max_edsc)
        star.solve_structure(self.prior.EOS.energydensities,
                                 self.prior.EOS.pressures)
        if(star.Mrot < 1):
                return -1e101

        for i in range(self.prior.number_stars):
            star = Star(10**(pr_dict['rhoc_' + str(i + 1)]), 0.0)
            star.solve_structure(self.prior.EOS.energydensities,
                                 self.prior.EOS.pressures)
            if(star.Mrot < 1.):
                return -1e101
            
        loglike_const = 1.
        return loglike_const