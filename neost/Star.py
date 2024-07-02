from matplotlib import pyplot
import warnings

try:
    from neost.tovsolvers.TOVr import solveTOVr
    from neost.tovsolvers.TOVh import solveTOVh
except ImportError:
    from neost.tovsolvers.TOVr_python import solveTOVr
    warnings.warn('Something is wrong with the C TOV solvers, using Python TOV solver instead. This is much slower.')
from . import global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi


class Star():
    """
    Instances of this class represent a model neutron star.
    """

    def __init__(self, epscent, enthalpy=False):
        """
        Initialize a model star.

        Args:
            epscent (float): The central energy density of the
            model star in cgs units.

        """
        self.epscent = epscent
        self.Mb = 0
        self.Rns = 0
        self.tidal = 0
        self.enthalpy = enthalpy

    def solve_structure(self, eps, pres, atol=1e-6, rtol=1e-4, hmax=1000., step=0.46):
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
            self.Mb, self.Rns, self.tidal = solveTOVh(self.epscent,eps, pres)
            
        else:
                self.Mb, self.Rns, self.tidal, self.Gtt = solveTOVr(self.epscent, eps, pres, atol, rtol, hmax, step)


    @property
    def Mrot(self):
        """ Get the gravitational mass. """
        return (self.Mb) / Msun 


    @property
    def Req(self):
        """ Get the equatorial radius. """
        return (self.Rns) / 1e5



    def plot_structure(self):
        """
            Plot the internal mass and pressure structure of the star.
        """

        fig, ax = pyplot.subplots(1,2, figsize=(14,6))
        ax[0].plot(self.radius_grid_baryon, self.mass_dist_baryon,
                   c='#005ABD', lw=2.5, label='Baryonic')
        ax[1].plot(self.radius_grid_baryon, self.pres_dist_baryon, c='#005ABD',
                   lw=2.5, label='Baryonic')
        ax[1].set_yscale('log')

        ax[1].set_ylim(1e18, 1e36)
        ax[1].set_ylabel('p [dyn/cm$^{2}$]', fontsize=16)
        ax[0].set_ylabel('m [M$_{\odot}$]', fontsize=16)
        from matplotlib.ticker import ScalarFormatter

        for i in range(2):
            ax[i].tick_params(axis='both', which='major', labelsize=14)
            ax[i].set_xlabel('r [km]', fontsize=16)
            ax[i].set_xscale('log')
            ax[i].set_xlim(0.9, 150)
            ax[i].xaxis.set_major_formatter(ScalarFormatter())
        ax[0].legend(prop={'size':16})
        pyplot.tight_layout()
        pyplot.show()
