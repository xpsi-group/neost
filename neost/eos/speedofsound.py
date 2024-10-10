import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import optimize
from scipy.integrate import cumulative_trapezoid, solve_ivp

from . base import BaseEoS                              ############## CAUTION: added modifications to n3lo gaussian sampling index

from .. import global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons
dyncm2_to_MeVfm3 = global_imports._dyncm2_to_MeVfm3
gcm3_to_MeVfm3 = global_imports._gcm3_to_MeVfm3
oneoverfm_MeV = global_imports._oneoverfm_MeV
n_ns = global_imports._n_ns


class SpeedofSoundEoS(BaseEoS):

    """
    Class representing an equation of state object.


    Parameters
    ----------
    eos_name : str
        The name of the EOS parameterization. Should be one of 'polytropes',
        'speedofsound' or 'spectral'.
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
    update(eos_params, max_edsc=True)
        Update the EoS object with a given set of parameters
    get_eos_crust()
        Construct the crust of the equation of state, with or without cEFT.
    get_eos()
        Construct the high-density parameterization of the equation of state.
    plot()
        Plot the equation of state.
    plot_massradius()
        Plot the mass-radius curve of the equation of state.

    """

    def __init__(self, crust, rho_t):

        super(SpeedofSoundEoS, self).__init__(crust, rho_t)

        self.eos_name = 'speedofsound'
        self.param_names = ['a1', 'a2', 'a3/a2', 'a4', 'a5']
        if self.ceft is True:
            self.param_names.append('ceft')
            self.param_names.append('ceft_in')                 ### added


    def get_eos(self):

        self._eds_core = np.logspace(np.log10(self.eds_t), 16.65, 400)
        cs_t = self.CofE(self.eds_t, self._eds_crust, self._pres_crust)
        sol = optimize.minimize(self.match_func, [0.],
                                bounds=[(-5., 5.)], args=(cs_t))
        self.norm = sol.x

        self._pres_core = (cumulative_trapezoid(self.Cs_model_core(self._eds_core / rho_ns,
                           self.norm, negative=0.0), self._eds_core,
                           initial=0.0) * c**2. + self.P_t)

        result = solve_ivp(lambda eps, rho: self.rhodens(rho, eps),
                           t_span=(self._eds_core[0], self._eds_core[-1]),
                           y0=[self.rho_t], t_eval=self._eds_core,
                           method='LSODA')

        self._rho_core = result.y[0]

        totalrho = np.hstack([self._rho_crust, self._rho_core[1:]])
        totalpres = np.hstack([self._pres_crust, self._pres_core[1:]])
        totaleps = np.hstack([self._eds_crust, self._eds_core[1:]])
        self.pressures = totalpres #orginally in cgs 
        self.energydensities = totaleps 
        self.massdensities = totalrho 

        self.eos = UnivariateSpline(self.energydensities,
                                    self.pressures, k=1, s=0)


    #######################
    # Auxiliary functions #
    #######################

    # Speed of sound model
    def logistic(self, x, norm=0.0, xt=5, s=1.):
        return norm + (1. / 3. - norm) / (1. + np.exp(-s * (x - xt)))

    def Cs_model_core(self, x, norm, negative=-1e100):
        mean = self.eos_params['a2']
        sigma = self.eos_params['a3/a2'] * self.eos_params['a2']
        gauss = self.eos_params['a1'] * np.exp(-.5 * (x - mean)**2. /
                                                  sigma**2.)
        back = self.logistic(x, norm, xt=self.eos_params['a4'],
                             s=self.eos_params['a5'])
        return np.clip(back + gauss, a_min=negative, a_max=None)

    def match_func(self, x, cscrust):
        return (self.Cs_model_core(self.eds_t/rho_ns, x, negative=-1e100) -
                cscrust)**2.

    def Cs_model_total(self, x, norm):
        xt = self.rho_t / rho_ns
        beta = 1e-10
        dmin = .5 * (1. - np.tanh(pi / beta * (x - xt)))
        dplus = .5 * (1. + np.tanh(pi / beta * (x - xt)))

        return (dmin * self._cs_crust(x * rho_ns) / c**2. +
                dplus * (self.Cs_model_core(x, norm)))


    def check_constraints(self):
        check = True

        FermiCrit = (3. * ((3. * pi**2. * 1.5 * n_ns *
                     (197.33)**3.)**(1. / 3.))**2. / (3. * 939.565**2.))

        csFermi = self.Cs_model_total(self.EofRho(1.5 * rho_ns,
                                      self.massdensities,
                                      self.energydensities) /
                                      rho_ns, self.norm) 

        if csFermi > FermiCrit or np.any(self.Cs_model_core(self._eds_core / rho_ns, self.norm, negative=0.0) > 1.0):
            check = False

        tmp = np.linspace(50, self.eos_params['a2'], 100)
        rising = np.where(np.diff(self.Cs_model_core(tmp, self.norm)) > 0.)[0]
        falling = np.where(np.diff(self.Cs_model_core(tmp, self.norm)) < 0.)[0]

        if falling.size == 0:
            check = False
        if rising.size != 0 and falling.size != 0 and rising[0] < falling[0]:
            check = False

        return check
