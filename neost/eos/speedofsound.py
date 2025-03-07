import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import optimize
from scipy.integrate import cumulative_trapezoid, solve_ivp

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
        Can be one of 'Hebeler', 'Drischler', 'Lynn', 'Keller-N2LO', 'Keller-N3L0', 'Goettling-N2LO', 'Goettling-N3L0'  or 'Tews' (or 'old').
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
    add_adm_eos()
        Function to compute the ADM EoS, whether it be bosonic or fermionic in nature.
    plot()
        Plot the equation of state.
    plot_massradius()
        Plot the mass-radius curve of the equation of state.

    """

    def __init__(self, crust, rho_t, adm_type = 'None', dm_halo = False, two_fluid_tidal = False):

        super(SpeedofSoundEoS, self).__init__(crust, rho_t)

        self.eos_name = 'speedofsound'
        self.param_names = ['a1', 'a2', 'a3/a2', 'a4', 'a5']

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

        self._eds_core = np.logspace(np.log10(self.eds_t), 16.65, 400)
        cs_t = self.CofE(self.eds_t, self._eds_crust, self._pres_crust)
        sol = optimize.minimize(self.match_func, [0.],
                                bounds=[(-5., 5.)], args=(cs_t))
        self.norm = sol.x

        self._pres_core = (cumulative_trapezoid(self.Cs_model_core(self._eds_core / rho_ns,
                           self.norm, negative=0.0), self._eds_core,
                           initial=0.0) * c**2. + self.P_t)

        if self.crust == 'ceft-Goettling-N2LO' or self.crust == 'ceft-Goettling-N3LO':
            result = solve_ivp(lambda eps, rho: self.rhodens(rho, eps),
                           t_span=(self._eds_core[0], self._eds_core[-1]),
                           y0=[self.Rho_t], t_eval=self._eds_core,
                           method='LSODA')                                                        #Rho_t coming from base.py instead of rho_t input by user, to avoid artificial phase transition
        else:
            result = solve_ivp(lambda eps, rho: self.rhodens(rho, eps),
                           t_span=(self._eds_core[0], self._eds_core[-1]),
                           y0=[self.rho_t], t_eval=self._eds_core,
                           method='LSODA')

        self._rho_core = result.y[0]

        totalrho = np.hstack([self._rho_crust, self._rho_core[1:]])
        totalpres = np.hstack([self._pres_crust, self._pres_core[1:]])
        totaleps = np.hstack([self._eds_crust, self._eds_core[1:]])
        self.pressures = totalpres 
        self.energydensities = totaleps 
        self.massdensities = totalrho 

        self.eos = UnivariateSpline(self.energydensities,
                                    self.pressures, k=1, s=0)

        if self.adm_type in ['Bosonic', 'Fermionic']:
            self.add_adm_eos()


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
        if self.crust == 'ceft-Goettling-N2LO' or self.crust == 'ceft-Goettling-N3LO':
            xt = self.Rho_t / rho_ns
        else:
            xt = self.rho_t / rho_ns
        beta = 1e-10
        dmin = .5 * (1. - np.tanh(pi / beta * (x - xt)))
        dplus = .5 * (1. + np.tanh(pi / beta * (x - xt)))

        return (dmin * self._cs_crust(x * rho_ns) / c**2. +
                dplus * (self.Cs_model_core(x, norm)))


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
            
        ### we could build another check_constraints function in base.py, just for the crust with normal distribution of cEFT, but we won't now
        ### it could maybe speed up the sampling? we check later
        
        if self.crust == 'ceft-Goettling-N2LO' or self.crust == 'ceft-Goettling-N3LO':
            if all(x<=y for x, y in zip(self._pres_crust, self._pres_crust[1:]))==False:   ## self._pres_crust might be overkill (includes low, BPS and cEFT)
                check = False
            else:
                check = True     

        return check
