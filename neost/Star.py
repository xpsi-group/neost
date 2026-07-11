import matplotlib.pyplot as plt
import warnings
import numpy as np

try:
    from neost.tovsolvers.TOVr import solveTOVr
    from neost.tovsolvers.TOVh import solveTOVh
    from neost.tovsolvers.TOVdm import solveTOVdm
except ImportError:
    from neost.tovsolvers.TOVr_python import solveTOVr
    from neost.tovsolvers.TOVdm_python import solveTOVdm
    warnings.warn('C TOV solvers either not installed or broken, using Python TOV solver instead. This is much slower.')
from . import global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi


class Star():
    """
    Instances of this class represent a model neutron star.
    """


    def __init__(self, epscent,epscent_dm = 0.0, enthalpy=False):
        """
        Initialize a model star.

        Args:
            epscent (float): The central energy density of the
            model star in cgs units.

            epscent_dm (float): The central energy density of the
            dark matter component in cgs units

        """
        self.epscent = epscent
        self.epscent_dm = epscent_dm
        self.Mb = 0
        self.Rns = 0
        self.tidal = 0
        self.enthalpy = enthalpy
        self.radius_dm_core = 0
        self.radius_dm_halo = 0
        self.Mdm_core = 0
        self.Mdm_halo = 0

    def solve_structure(self, eps, pres,eps_dm = None,pres_dm = None,dm_halo = False,two_fluid_tidal = False, atol=1e-6, rtol=1e-4, hmax=1000., step=0.46, rot_cor = None, freq = None):
        """
        Solve the relativistic structure equations to build a model star.

        Args:
            :param eos (object): The interpolated EoS object, which inputs a density
                          and outputs a pressure, both in geometrized units.
            :param Pmin (float): The pressure at the surface of the star, or the
                          minimum pressure for which the EoS is defined.
                          Should be given in geometrized units.
            :param atol (float): The absolute tolerance for the ODE solver.
            :param rtol (float): The relative tolerance for the ODE solver.
            :param hmax (float): The maximum step size allowed for the ODE solver,
                          given in centimetres.
            :param step (float): The resolution of the ODE solver,
                          should be between ~0.2 and ~0.5.


        """

        if self.enthalpy is True:
            if self.epscent_dm == 0.0:
                self.Mb, self.Rns, self.tidal = solveTOVh(self.epscent,eps, pres)
            else:
                raise Exception("two-fluid enthalpy solve currently not available")
               #two-fluid enthalpy tov solver not available currently
               #self.Mb, self.Rns, self.tidal, self.Mdm_core, self.radius_dm_core = solveTOVdmh(self.epscent, self.epscent_dm, eps, pres, eps_dm, pres_dm)

        else:
            if self.epscent_dm == 0.0:
                self.Mb, self.Rns, self.tidal, self.Gtt = solveTOVr(self.epscent, eps, pres, atol, rtol, hmax, step)

            else:
                self.Mb, self.Rns, self.Mdm_core, self.Mdm_halo, self.radius_dm_core, self.radius_dm_halo, self.tidal = solveTOVdm(self.epscent, self.epscent_dm, eps, pres, eps_dm, pres_dm, dm_halo,two_fluid_tidal, atol, rtol, hmax, step)
        if freq != 0:
            if rot_cor == 'Uni_rel':
           
                Ar=0.20321413068003585
                Br=0.16109256126681384
                Am=1.1273275369099987
                ar1=-15.495625337227967
                ar2=442.60235040521815
                ar3=-4945.622485975098
                ar4=23458.06339674752
                ar5=-40544.25802438619
                am0=-0.016001484791546946
                am1=3.1226208303364116
                am2=-20.720628851199855
                am3=41.202467878903896
                am4=-6.464335487164061
                a0=0.5522875476569138
                a1=3.304152584215269
                a2=-35.21127224435375
                a3=180.6110095901739
                a4=-326.4761248935815
        

                # Calculate the angular velocity
                omega = 2 * pi * freq
         
                #calculate the relative magnitude of the rotations compared to the kepler limit
                com = self.Mb/Msun*(1e5/self.Rns)
                omega_star= (G * self.Mb / self.Rns**3)**0.5
                omega_k  = omega_star * (a0 * com**0 +a1 * com**1 + a2 * com**2 + a3 *com**3 + a4 * com**4)

                OMEGA = omega/omega_k

                #if rotation is too far above kepler limit the rotation is deemed impossible 
                #we use 1.1 in stead of 1 because this is an empirical model which can have some deviation from the actual values
                if OMEGA >= 1.:
                    self.Rns = np.nan
                    self.Mb = np.nan
                else:
                    #formulas for the universal relations taken from https://arxiv.org/pdf/2206.12515
                    self.Rns = self.Rns * (1 + (np.exp(Ar*OMEGA**2) - 1 + Br*np.log(1-(OMEGA/1.1)**4)**2)*(1 + ar1 * com**1 + ar2 * com**2 +ar3 * com**3 + ar4 * com**4 +ar5 * com**5))
                
                    self.Mb = self.Mb * (1 + (np.exp(Am * OMEGA**2) - 1) * (am0 * com**0 + am1 * com + am2 * com**2 + am3 * com**3 + am4 * com**4) )




    @property
    def Mrot(self):
        """ Get the gravitational mass. """
        return (self.Mb + self.Mdm_core + self.Mdm_halo) / Msun

    @property
    def Mbaryonic(self):
        """ Get the gravitational mass of baryonic component. """
        return (self.Mb)/ Msun

    @property
    def Req(self):
        """ Get the equatorial radius. """
        return (self.Rns) / 1e5

    @property
    def Mdm(self):
        """ Get the total gravitational dark matter mass. """
        return (self.Mdm_core + self.Mdm_halo) / Msun

    @property
    def Mdmcore(self):
        """ Get the dark matter mass within the baryonic radius. """
        return (self.Mdm_core) / Msun

    @property
    def Mdmhalo(self):
        """ Get the dark matter mass outside the baryonic radius. """
        return (self.Mdm_halo) / Msun

    @property
    def Rdm(self):
        """ Get the total dark matter radius."""
        if self.radius_dm_halo == 0 or self.radius_dm_halo == 999e5:
            Rdm = self.radius_dm_core

        else:
            Rdm = self.radius_dm_halo
        return (Rdm) / 1e5

    @property
    def Rdm_core(self):
        """ Get the dark matter core radius. """
        return self.radius_dm_core  / 1e5

    @property
    def Rdm_halo(self):
        """ Get the dark matter halo radius. """
        return self.radius_dm_halo  / 1e5

    def plot_structure(self):
        """
            Plot the internal mass and pressure structure of the star.
        """

        radius_grid_baryon = self.dist_baryon[:,0]/1e5
        mass_dist_baryon = self.dist_baryon[:,1]/ Msun
        pres_dist_baryon = self.dist_baryon[:,2]
        fig, ax = pyplot.subplots(1,2, figsize=(14,6))
        ax[0].plot(radius_grid_baryon, mass_dist_baryon,
                   c='#005ABD', lw=2.5, label='Baryonic')
        ax[1].plot(radius_grid_baryon, pres_dist_baryon, c='#005ABD',
                   lw=2.5, label='Baryonic')
        ax[1].set_yscale('log')

        if self.epscent_dm!=0:
            radius_grid_dm = self.dist_dm[:,0]/1e5
            mass_dist_dm = self.dist_dm[:,1]/ Msun
            pres_dist_dm = self.dist_dm[:,2]
            ax[0].plot(radius_grid_dm, mass_dist_dm, c='#00B1B7', lw=2.5,
                   label='Dark matter')
            ax[1].plot(radius_grid_dm,pres_dist_dm, c='#00B1B7',
                   lw=2.5, label='Dark matter')

        ax[1].set_ylim(1e18, 1e36)
        ax[1].set_ylabel('p [dyn/cm$^{2}$]', fontsize=16)
        ax[0].set_ylabel(r'm [M$_{\odot}$]', fontsize=16)
        from matplotlib.ticker import ScalarFormatter

        for i in range(2):
            ax[i].tick_params(axis='both', which='major', labelsize=14)
            ax[i].set_xlabel('r [km]', fontsize=16)
            ax[i].set_xscale('log')
            ax[i].set_xlim(0.9, 150)
            ax[i].xaxis.set_major_formatter(ScalarFormatter())
        ax[0].legend(prop={'size':16})

        pyplot.savefig('Structure_plot.png',dpi = 300)
        plt.tight_layout()
        plt.show()
