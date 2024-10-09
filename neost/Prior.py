import numpy as np

from neost import global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons


# TODO: put a constraint that the pr and hypercube must
# be same length as bounds


class Prior():

    def __init__(self, EOS, variable_params, static_params, chirp_masses):

        self.variable_params = variable_params
        self.EOS = EOS
        self.static_params = static_params
        self.number_stars = len(chirp_masses)
        self.chirp_masses = chirp_masses
        self.prior_names = self.EOS.param_names.copy()
        for i in range(self.number_stars):
            self.prior_names.append('rhoc_' + str(i + 1))

    def inverse_sample(self, hypercube):
        hypercube = {e:hypercube[i] for i, e in
                     enumerate(self.variable_params)}
        pr = {e:hypercube[e] *
              (self.variable_params[e][1] - self.variable_params[e][0]) +
              self.variable_params[e][0] for i, e in
              enumerate(list(self.variable_params.keys()))}

        if 'rho_t1' and 'rho_t2' in self.variable_params.keys():
            # forced identifiability prior #
            pr['rho_t1'] = ((1. - np.sqrt(hypercube['rho_t1'])) *
                            (self.variable_params['rho_t1'][1] - 
                            self.variable_params['rho_t1'][0]) +
                            self.variable_params['rho_t1'][0])

            pr['rho_t2'] = ((1. - hypercube['rho_t2']) *
                            (self.variable_params['rho_t2'][1] - pr['rho_t1'])
                            + pr['rho_t1'])

            if 'mchi' in self.variable_params.keys():
                pr['mchi'] = 10**pr['mchi']

            if 'gchi_over_mphi' in self.variable_params.keys():
                pr['gchi_over_mphi' ] = 10**pr['gchi_over_mphi']

        pr.update(self.static_params)
        self.EOS.update({k: pr[k] for k in tuple(self.EOS.param_names)},
                        max_edsc=True)
        self.pr = pr

        for i in range(self.number_stars):
            if self.chirp_masses[i] is None:
                logminedsc = np.log10(self.EOS.min_edsc)
                logmaxedsc = np.log10(self.EOS.max_edsc)
                pr.update({'rhoc_' + str(i + 1):hypercube['rhoc_' + str(i + 1)]
                          * (logmaxedsc - logminedsc) + logminedsc})
            else:
                logminedsc, logmaxedsc = self.EOS.get_minmax_edsc_chirp(
                    self.chirp_masses[i])
                logminedsc = np.log10(logminedsc)
                logmaxedsc = np.log10(logmaxedsc)
                pr.update({'rhoc_' + str(i + 1):hypercube['rhoc_' + str(i + 1)]
                          * (logmaxedsc - logminedsc) + logminedsc})
        self.MRT = self.EOS.massradius
        self.max_edsc = self.EOS.max_edsc

        return list(pr.values())
