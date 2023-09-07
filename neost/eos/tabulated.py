import numpy as np
from math import pow
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp

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


class TabulatedEoS(BaseEoS):

    """
    Class representing a tabulated equation of state object.


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

    def __init__(self, energydensity, pressure, crust=None, rho_t=None):

        super(TabulatedEoS, self).__init__(crust, rho_t)

        self.eos_name = 'tabulated'
        self.param_names = []

        self.energydensities = energydensity # Assumed to be in cgs (g/cm^3)
        self.pressures = pressure  #Assumed to be in g/(cm s^2)

    def get_eos(self):

        eps0 = self.energydensities[0]
        self._eds_core = self.energydensities
        self._pres_core = self.pressures
        result = solve_ivp(lambda eps, rho: self.rhodens(rho, eps), t_span=(self._eds_core[0], self._eds_core[-1]),
                           y0=[eps0], t_eval=self._eds_core, method='LSODA')
        self.massdensities = result.y[0]

        self.eos = UnivariateSpline(self.energydensities, self.pressures, k=1, s=0)




    def check_constraints(self):
        check = True #required because there are checks in speedofsound.py file, but no constraints are needed for this file
        return check


