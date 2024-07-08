import numpy as np
from math import pow
from scipy.interpolate import UnivariateSpline,interp1d
from scipy.integrate import odeint

from . base import BaseEoS

from .. import global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons
dyncm2_to_MeVfm3 = global_imports._dyncm2_to_MeVfm3
gcm3_to_MeVfm3 = global_imports._gcm3_to_MeVfm3
oneoverfm_MeV = global_imports._oneoverfm_MeV


class PolytropicEoS(BaseEoS):

    """
    Class representing a polytropic equation of state object.


    Parameters
    ----------
    rho_t: float
        The transition density between the crust EOS and the high density
        parameterization in cgs.
    ceft: bool
        If True a low-density cEFT parameterization is used.
    ceft_method: str
        The name of the cEFT calculations used at low density.
        Can be one of 'Hebeler', 'Drischler', 'Lynn' or 'Tews'.

    Methods
    -------
    get_eos()
        Construct the high-density parameterization of the equation of state.
    eos_core_pp()
        Function to compute the polytropic equation of state parameterization.

    """

    def __init__(self, crust, rho_t):

        super(PolytropicEoS, self).__init__(crust, rho_t)

        self.eos_name = 'polytropes'
        self.param_names = ['gamma1', 'gamma2', 'gamma3',
                            'rho_t1', 'rho_t2']
 

        if self.ceft is True:
            self.param_names.append('ceft')


    def get_eos(self):

        self.gammas = list(map(self.eos_params.get,
                               ['gamma1', 'gamma2', 'gamma3']))
        self.rho_ts = list(map(self.eos_params.get, ['rho_t1', 'rho_t2']))
        self._rho_core = np.zeros(297)

        self._rho_core[0:99] = np.linspace(self.rho_t / rho_ns,
                                           self.rho_ts[0], 100)[1::]
        self._rho_core[99:198] = np.linspace(self.rho_ts[0],
                                             self.rho_ts[1], 100)[1::]
        self._rho_core[198::] = np.logspace(np.log10(self.rho_ts[1]),
                                            2.2, 100)[1::]

        self._pres_core = np.zeros(len(self._rho_core))
        for i, e in enumerate(self._rho_core):
            self._pres_core[i] = self.eos_core_pp(e, 
                                                  self.P_t * dyncm2_to_MeVfm3)

        totalrho = np.hstack([self._rho_crust, self._rho_core * rho_ns])
        totalpres = np.hstack([self._pres_crust,
                               self._pres_core / dyncm2_to_MeVfm3])

        eps0 = self._eds_crust[-1]
        prho = 0
        totalrho,indicies = np.unique(totalrho,return_index = True)
        totalpres = totalpres[indicies]
        try:
            prho = UnivariateSpline(totalrho, totalpres, k=2, s=0)
        except ValueError:
            prho = interp1d(totalrho,totalpres,kind = 'linear',fill_value = 'extrapolate')

        result = odeint(self.edens, eps0,
                        totalrho[totalrho >= self._rho_crust[-1]],
                        args=tuple([prho]))
        self._eds_core = result.flatten()[1::]

        totaleps = np.hstack([self._eds_crust, self._eds_core])
        self.pressures = totalpres #oringally in cgs units
        self.energydensities = totaleps   
        self.massdensities = totalrho

        self.eos = UnivariateSpline(self.energydensities,
                                    self.pressures, k=1, s=0)



    def eos_core_pp(self, rho, P_t):
        P_ts, k = (np.zeros(len(self.gammas)) for i in range(2))
        P_ts[0] = P_t
        k[0] = P_t / ((self.rho_t / rho_ns)**self.gammas[0])
        P_ts[1] = k[0] * self.rho_ts[0]**self.gammas[0]
        k[1] = P_ts[1] / (self.rho_ts[0]**self.gammas[1])
        P_ts[2] = k[1] * self.rho_ts[1]**self.gammas[1]
        k[2] = P_ts[2] / (self.rho_ts[1]**self.gammas[2])
        self.pres_ts = P_ts[1::] / dyncm2_to_MeVfm3

        if rho <= self.rho_ts[0]:
            pres = k[0] * rho**self.gammas[0]

        if self.rho_ts[0] < rho <= self.rho_ts[1]:
            pres = k[1] * rho**self.gammas[1]

        if self.rho_ts[1] < rho:
            pres = k[2] * rho**self.gammas[2]

        return pres

    def polytropic_func(self, rho, K, index):
        return K * rho**index

    def check_constraints(self):
        check = True #required because there are checks in speedofsound.py file, but no constraints are needed for this file
        return check

