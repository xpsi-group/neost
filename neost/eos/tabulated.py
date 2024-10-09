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
        Can be one of 'Hebeler', 'Drischler', 'Lynn', 'Keller-N2LO', 'Keller-N3L0', or 'Tews'.
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
    get_eos()
        Construct the equation of state.
    add_adm_eos()
        Function to compute the ADM EoS, whether it be bosonic or fermionic in nature.
    plot()
        Plot the equation of state.
    plot_massradius()
        Plot the mass-radius curve of the equation of state.

    """

    def __init__(self, energydensity, pressure, crust=None, rho_t=None,adm_type = 'None',dm_halo = False,two_fluid_tidal = False):

        super(TabulatedEoS, self).__init__(crust, rho_t)

        self.adm_type = adm_type
        self.dm_halo = dm_halo
        self.two_fluid_tidal = two_fluid_tidal

        self.eos_name = 'tabulated'
        self.param_names = []

        self.energydensities = energydensity # Assumed to be in cgs (g/cm^3)
        self.pressures = pressure  #Assumed to be in g/(cm s^2)
        
        if self.adm_type not in ['Bosonic', 'Fermionic', 'None']:
            raise TypeError('ADM model not recognized. Choose either "Bosonic" or "Fermionic". If ADM is not needed, select "None".')
        
        if self.adm_type in ['Bosonic','Fermionic']:
            
            self.param_names +=['mchi','gchi_over_mphi', 'adm_fraction']

    def get_eos(self):

        eps0 = self.energydensities[0]
        self._eds_core = self.energydensities
        self._pres_core = self.pressures
        result = solve_ivp(lambda eps, rho: self.rhodens(rho, eps), t_span=(self._eds_core[0], self._eds_core[-1]),
                           y0=[eps0], t_eval=self._eds_core, method='LSODA')
        self.massdensities = result.y[0]

        self.eos = UnivariateSpline(self.energydensities, self.pressures, k=1, s=0)
        if self.adm_type in ['Bosonic', 'Fermionic']:
            self.add_adm_eos()




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
       check = True
       if self.reach_fraction == True:
           check = True
       else:
           check = False
       return check


