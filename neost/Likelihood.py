import numpy as np
from scipy.interpolate import UnivariateSpline
import warnings

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
            if self.prior.EOS.adm_type == 'Bosonic':
                epsdm_cent = self.prior.EOS.find_epsdm_cent(ADM_fraction=pr_dict['adm_fraction'],
                                                            epscent = 10**(pr_dict['rhoc_' + str(i + 1)]))
                if (self.prior.EOS.reach_fraction == False):
                        return -1e101

                elif (self.prior.EOS.reach_fraction == True):
                    star = Star(10**(pr_dict['rhoc_' + str(i + 1)]), epsdm_cent)
                    star.solve_structure(self.prior.EOS.energydensities,
                                     self.prior.EOS.pressures,
                                     self.prior.EOS.energydensities_dm,
                                     self.prior.EOS.pressures_dm,
                                     self.prior.EOS.dm_halo,
                                     self.prior.EOS.two_fluid_tidal)
                    Mgrav = star.Mrot
                    Req = star.Req
                    Rdm_halo = star.Rdm_halo
                    tidal = star.tidal
                #print('adm: ', Mgrav, Req, Rdm_halo,np.log10(pr_dict['mchi']), np.log10(pr_dict['gchi_over_mphi']), pr_dict['adm_fraction'])
                
            if self.prior.EOS.adm_type == 'Fermionic':
                #Hard cut-off imposed as all stars within these boxes have masses well below 1 Msun [~0.4 Msun down to ~0.001 Msun], thus this will save computation time if the code doesn't even have to compute them.
                if (pr_dict['mchi'] >= pow(10,6) and pr_dict['gchi_over_mphi'] <= pow(10,-3.5) and pr_dict['adm_fraction'] >= 0.01):
                    return -1e101
                elif (pr_dict['mchi'] >= pow(10,7) and pr_dict['gchi_over_mphi'] <= pow(10,-3) and pr_dict['adm_fraction'] >= 0.01):
                    return -1e101
                elif (pr_dict['mchi'] >= pow(10,8) and pr_dict['gchi_over_mphi'] <= pow(10,-2) and pr_dict['adm_fraction'] >= 0.01):
                    return -1e101
                elif (pr_dict['mchi'] >= pow(10,8.5) and pr_dict['gchi_over_mphi'] <= pow(10,-1) and pr_dict['adm_fraction'] >= 0.01):
                    return -1e101
                else:
                    epsdm_cent = self.prior.EOS.find_epsdm_cent(ADM_fraction=pr_dict['adm_fraction'],epscent = 10**(pr_dict['rhoc_' + str(i + 1)]))
                    if (self.prior.EOS.reach_fraction == False):
                        return -1e101

                    elif (self.prior.EOS.reach_fraction == True):
                        star = Star(10**(pr_dict['rhoc_' + str(i + 1)]), epsdm_cent)
                        star.solve_structure(self.prior.EOS.energydensities, 
                                            self.prior.EOS.pressures,
                                            self.prior.EOS.energydensities_dm,
                                            self.prior.EOS.pressures_dm,
                                            self.prior.EOS.dm_halo,
                                            self.prior.EOS.two_fluid_tidal)
                
                        Mgrav = star.Mrot
                        Req = star.Req
                        Rdm_halo = star.Rdm_halo
                        tidal = star.tidal
                    
                
            if self.prior.EOS.adm_type == 'None':
                star = Star(10**(pr_dict['rhoc_' + str(i + 1)]), 0.0)
                star.solve_structure(self.prior.EOS.energydensities,self.prior.EOS.pressures)
                Mgrav = star.Mrot
                Req = star.Req
                tidal = star.tidal
                Rdm_halo = 0.0
                
                

            if (Mgrav > 3. or Mgrav < 1. or Req > 16. or Req < Mgrav / Msun * 2. * G / c**2. / 1e5): 
                return -1e101

            if self.prior.EOS.dm_halo == False and Rdm_halo > 0:
                return -1e101


            if self.chirp_masses[i] is None:
                MassRadiusTidal = {'Mass':Mgrav, 'Radius':Req,
                                   'Lambda':tidal}
                tmp = list(map(MassRadiusTidal.get, self.likelihood_params[i]))
                like = self.likelihood_functions[i](tmp)
                likelihoods.append(self.array_to_scalar(like))
            else:
                if self.prior.EOS.adm_type != 'None':
                    warnings.warn("Two-fluid tidal deformability only implemented in python. Performance will slow!!")

                M2 = Mgrav
                M1 = m1_from_mc_m2(self.chirp_masses[i], M2)
                if M1 < M2 or M1 > self.prior.MRT[:,0][-1]:
                    return -1e101
                MTspline = UnivariateSpline(self.prior.MRT[:,0],
                                            self.prior.MRT[:,2], k=1, s=0)
                point = np.array([self.chirp_masses[i], M2 / M1,
                                    MTspline(M1), tidal])
                like = self.likelihood_functions[i](point)
                likelihoods.append(self.array_to_scalar(like))

        
        like_total = np.prod(np.array(likelihoods))
        # print('lnlike is', like_total)
        if like_total == 0.0:
            return -1e101

        return np.log(like_total)
    
    def array_to_scalar(self, var):
        # np no longer accepts ragged arrays,
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

        if self.prior.EOS.adm_type == 'None':
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

        if self.prior.EOS.adm_type == 'Bosonic' or self.prior.EOS.adm_type == 'Fermionic':
            if (pr_dict['mchi'] >= pow(10,6) and pr_dict['gchi_over_mphi'] <= pow(10,-3.5) and pr_dict['adm_fraction'] >= 0.01):
                return -1e101
            elif (pr_dict['mchi'] >= pow(10,7) and pr_dict['gchi_over_mphi'] <= pow(10,-3) and pr_dict['adm_fraction'] >= 0.01):
                return -1e101
            elif (pr_dict['mchi'] >= pow(10,8) and pr_dict['gchi_over_mphi'] <= pow(10,-2) and pr_dict['adm_fraction'] >= 0.01):
                return -1e101
            elif (pr_dict['mchi'] >= pow(10,8.5) and pr_dict['gchi_over_mphi'] <= pow(10,-1) and pr_dict['adm_fraction'] >= 0.01):
                return -1e101
            else:
                epsdm_cent = self.prior.EOS.find_epsdm_cent(ADM_fraction=pr_dict['adm_fraction'],
                                                            epscent = self.prior.EOS.max_edsc)

                if self.prior.EOS.reach_fraction == False:
                    return -1e101

                star = Star(self.prior.EOS.max_edsc,epsdm_cent)
                star.solve_structure(self.prior.EOS.energydensities,
                                    self.prior.EOS.pressures,
                                    self.prior.EOS.energydensities_dm,
                                    self.prior.EOS.pressures_dm,
                                    self.prior.EOS.dm_halo,
                                    self.prior.EOS.two_fluid_tidal)
                if(star.Mrot < 1.):
                    return -1e101

                if self.prior.EOS.dm_halo == False and star.Rdm_halo > 0:
                    return -1e101

                for i in range(self.prior.number_stars):
                    epsdm_cent = self.prior.EOS.find_epsdm_cent(ADM_fraction=pr_dict['adm_fraction'],
                                                            epscent = 10**(pr_dict['rhoc_' + str(i + 1)]))

                    if self.prior.EOS.reach_fraction == False:
                        return -1e101

                    star = Star(10**(pr_dict['rhoc_' + str(i + 1)]), epsdm_cent)
                    star.solve_structure(self.prior.EOS.energydensities,
                                    self.prior.EOS.pressures,
                                    self.prior.EOS.energydensities_dm,
                                    self.prior.EOS.pressures_dm,
                                    self.prior.EOS.dm_halo,
                                    self.prior.EOS.two_fluid_tidal)
                    if(star.Mrot < 1.):
                        return -1e101

                    if self.prior.EOS.dm_halo == False and star.Rdm_halo > 0:
                        return -1e101

            
        loglike_const = 1.
        return loglike_const
