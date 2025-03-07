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
n_ns = global_imports._n_ns
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
        Can be one of 'Hebeler', 'Drischler', 'Lynn', 'Keller-N2LO', 'Keller-N3L0', 'Goettling-N2LO', 'Goettling-N3L0' or 'Tews' (or 'old').
    adm_type: str
        The name of the ADM particle type. Can be 'None', 'Bosonic', or 'Fermionic'
    dm_halo: bool
        If True, ADM halos will be allowed.
    two_fluid_tidal: bool
        If True, the two-fluid tidal deformability solver will be used. See TidalDef.py file

    Methods
    -------
    update(eos_params, max_edsc=True)
        Update the EoS object with a given set of parameters
    get_eos_crust()
        Construct the crust of the equation of state, with or without cEFT.
    get_eos_crust_GP()
        Construct the crust of the equation of state, with normal distribution of cEFT EOS. Needs 'Goettling-N2LO' or 'Goettling-N3L0'.
    get_eos()
        Construct the high-density parameterization of the equation of state.
    eos_core_pp()
        Function to compute the polytropic equation of state parameterization.
    add_adm_eos()
        Function to compute the ADM EoS, whether it be bosonic or fermionic in nature.
    plot()
        Plot the equation of state.
    plot_massradius()
        Plot the mass-radius curve of the equation of state.

    """

    def __init__(self, crust, rho_t, adm_type = 'None', dm_halo = False, two_fluid_tidal = False):

        super(PolytropicEoS, self).__init__(crust, rho_t)

        self.eos_name = 'polytropes'
        self.param_names = ['gamma1', 'gamma2', 'gamma3',
                            'rho_t1', 'rho_t2']

        self.adm_type = adm_type
        self.dm_halo = dm_halo
        self.two_fluid_tidal = two_fluid_tidal
 

        if self.ceft is True:
            self.param_names.append('ceft')

        if self.adm_type not in ['Bosonic', 'Fermionic', 'None']:
            raise TypeError('ADM model not recognized. Choose either "Bosonic" or "Fermionic". If ADM is not needed, select "None".')
        
        if self.adm_type in ['Bosonic','Fermionic']:
            
            self.param_names +=['mchi','gchi_over_mphi', 'adm_fraction']


    def get_eos(self):
        self.gammas = list(map(self.eos_params.get,
                               ['gamma1', 'gamma2', 'gamma3']))
        self.rho_ts = list(map(self.eos_params.get, ['rho_t1', 'rho_t2']))
        
        self._rho_core = np.zeros(297)

        if self.crust == 'ceft-Goettling-N2LO' or self.crust == 'ceft-Goettling-N3LO':
            self._rho_core[0:99] = np.linspace(self.Rho_t / rho_ns, self.rho_ts[0], 100)[1::]   #Rho_t coming from base.py instead of rho_t input by user, to avoid artificial phase transition
        else:
            self._rho_core[0:99] = np.linspace(self.rho_t / rho_ns, self.rho_ts[0], 100)[1::]
                        
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
        prho = 0     ## why?
        totalrho, indices = np.unique(totalrho,return_index = True)
        totalpres = totalpres[indices]
        try:
            prho = UnivariateSpline(totalrho, totalpres, k=2, s=0)
        except ValueError:
            prho = interp1d(totalrho, totalpres, kind = 'linear', fill_value = 'extrapolate')    ### why do we need this now?

        result = odeint(self.edens, eps0,
                        totalrho[totalrho >= self._rho_crust[-1]],
                        args=tuple([prho]))
        self._eds_core = result.flatten()[1::]

        totaleps = np.hstack([self._eds_crust, self._eds_core])
        self.pressures = totalpres 
        self.energydensities = totaleps   
        self.massdensities = totalrho

        self.eos = UnivariateSpline(self.energydensities,
                                    self.pressures, k=1, s=0)

        if self.adm_type in ['Bosonic', 'Fermionic']:
            self.add_adm_eos()

    def eos_core_pp(self, rho, P_t):
        P_ts, k = (np.zeros(len(self.gammas)) for i in range(2))
        P_ts[0] = P_t
        
        if self.crust == 'ceft-Goettling-N2LO' or self.crust == 'ceft-Goettling-N3LO':
            k[0] = P_t / ((self.Rho_t / rho_ns)**self.gammas[0])
        else:
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

    def add_adm_eos(self):

        def joinarray(ar1,ar2):
            return np.concatenate((ar1,ar2))
        
        _c = 2.99792458e8  # m/s
        _hbar = 1.0545718e-34  # J*s
        MeV_to_Joules = 1.60218e-13  # J

        self.mchi = self.eos_params['mchi'] * MeV_to_Joules / _c**2.
        self.gchi_over_mphi = (self.eos_params['gchi_over_mphi'] / MeV_to_Joules * _c**2.)
        self.adm_fraction = self.eos_params['adm_fraction']



        # Bosonic Dark matter component
        # geometric
        if self.adm_type == 'Bosonic':
            #print('Using bosonic EoS')
            pos_nchi = np.logspace(-5, 55, 5000)
            self.energydensities_dm = (self.mchi * _c**2 * pos_nchi + 1. / 2. *
                                       pow(self.gchi_over_mphi,2) * _hbar**3 *
                                       _c**(-1) * pos_nchi**2) *10 / c**2 #Factor of 10 is here to convert from SI base units to cgs base units
                                                                           # /c**2 is here since this is base units of g/(cm s^2) not g/cm^3
            self.pressures_dm = (1. / 2. * pow(self.gchi_over_mphi,2) *
                                 _hbar**3 * _c**(-1.) * pos_nchi**2) *10 #Factor of 10 is here to convert from SI base units to cgs base units

            self.massdensities_dm = pos_nchi * self.mchi * pow(10,-3) #Factor of 10^-3 is here to convert from SI base units to cgs base units

        if self.adm_type == 'Fermionic':
            #Fermionic Dark matter component
            # geometric units

            #print('Using fermionic EoS')
            pts = 5000 #default point value
  

            lower_bound = -2.5 #default lower bound

            upper_bound = 2.5 #default upper bound

            if self.eos_params['mchi'] < 10:
                upper_bound = 6
           
            if self.eos_params['mchi']>=pow(10,3):
                lower_bound  = -14 
        
                upper_bound = 1
                
            if self.eos_params['mchi']>= pow(10,5):
                lower_bound = -10
                upper_bound = 1
                

            x= np.logspace(lower_bound,upper_bound,pts)


            self.energydensities_dm = np.zeros(pts)

            self.pressures_dm = np.zeros(pts)

           
            # APPROXIMATION EoS FOR VALUES OF x < 10**2
            z = x[x<pow(10,-2)]
            interact_term = 1./2.*pow(self.gchi_over_mphi,2)*self.mchi**2*z**6/(3*np.pi**2)**2

            
            self.energydensities_dm[0:len(z)] = (_c**5*self.mchi**4/_hbar**3*(z**3/(3*np.pi**2) + interact_term))*10 / c**2 #Factor of 10 is here to convert from SI base units to cgs base unit
                                                                                                                     # /c**2 is here since this is base units of g/(cm s^2) not g/cm^3

            self.pressures_dm[0:len(z)] = (_c**5*self.mchi**4/_hbar**3*(z**5/(15*np.pi**2) + interact_term))*10 #Factor of 10 is here to convert from SI base units to cgs base unit

            # EXACT EoS FOR VALUES OF x >= 10**-2 
            y = x[x>=pow(10,-2)]
            const = (self.mchi**4*_c**5/(8*np.pi**2*_hbar**3))

            A= const*(2*y**3+y)*np.sqrt(1+y**2)

            B = const*np.log(y+np.sqrt(1+y**2))

            C = (1/(18*np.pi**4))*pow(self.gchi_over_mphi,2)*(_c**5/_hbar**3)*(self.mchi*y)**6

            self.energydensities_dm[len(z)::] = (A-B+C)*10 / c**2 #Factor of 10 is here to convert from SI base units to cgs base unit
                                                           # /c**2 is here since this is base units of g/(cm s^2) not g/cm^3

            self.pressures_dm[len(z)::] = (const*np.sqrt(1+y**2)*((2./ 3.*y**3)-y)+B+C)*10 #Factor of 10 is here to convert from SI base units to cgs base unit #10 ADDED ON 11/18. This is added to convert kg/ms^2 to g/cms^2

            nchi = 1./(3*np.pi**2)*(self.mchi*_c*x/_hbar)**3

            self.massdensities_dm = nchi * self.mchi * pow(10,-3) #Factor of 10^-3 is here to convert from SI base units to cgs base units


    def check_constraints(self):
        check = True                   #required because there are checks in speedofsound.py file, but no constraints are needed for this file, if no DM nor normal distribution of cEFT

        if self.reach_fraction == True:
            check = True
        else:
            check = False
            
        ### we could build another check_constraints function in base.py, just for the crust with normal distribution of cEFT, but we won't now
        ### it could maybe speed up the sampling? we check later
        
        if self.crust == 'ceft-Goettling-N2LO' or self.crust == 'ceft-Goettling-N3LO':
            if all(x<=y for x, y in zip(self._pres_crust, self._pres_crust[1:]))==False:   ## self._pres_crust might be overkill (includes low, BPS and cEFT)
                check = False
            else:
                check = True            
            
        return check

