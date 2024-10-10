import numpy
from scipy.interpolate import UnivariateSpline
from scipy import optimize
from scipy.integrate import odeint
from matplotlib import pyplot

from .. Star import Star
from .. import global_imports
from .. utils import m1_from_mc_m2, m1_m2_from_mc_q

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons
dyncm2_to_MeVfm3 = global_imports._dyncm2_to_MeVfm3
gcm3_to_MeVfm3 = global_imports._gcm3_to_MeVfm3
oneoverfm_MeV = global_imports._oneoverfm_MeV


class BaseEoS():                       #### CAUTION: important alterations hard-coded to make NLO bands run
                                       #### CAUTION: alterations in likelihood so that gaussian distribution in the ceft band, n3lo hard-coded
                                       #### further alterations to sample gaussian in indexes as well
    """
    Base class representing an equation of state object.


    Parameters
    ----------
    crust: str
        The name of the EoS crust model to use. Can be either 'ceft-Hebeler',
        'ceft-Drischler', 'ceft-Lynn', 'ceft-Tews', 'ceft-old', 'BPS', or None
        if a tabulated EoS with a crust model already included is used.
    rho_t: float
        The transition density between the crust EOS and the high density
        parameterization in cgs.

    Methods
    -------
    update(eos_params, max_edsc=True)
        Update the EoS object with a given set of parameters
    get_eos_crust()
        Construct the crust of the equation of state, with or without cEFT.
    plot()
        Plot the equation of state.
    plot_massradius()
        Plot the mass-radius curve of the equation of state.

    """

    def __init__(self, crust='ceft-Hebeler', rho_t=2e14):

        if crust not in ['ceft-Hebeler', 'ceft-Drischler', 'ceft-Lynn',
                         'ceft-Tews', 'ceft-Keller-LO', 'ceft-Keller-NLO', 'ceft-Keller-N2LO', 'ceft-Keller-N3LO', 'ceft-Keller-N3LO-MOD', 'ceft-old', 'BPS', None]:
            raise TypeError('crust model not recognized, choose either \
                "ceft-Hebeler", "ceft-Drischler", "ceft-Lynn", "ceft-Tews", \
                "ceft-Keller-LO", "ceft-Keller-NLO", "ceft-Keller-N2LO", "ceft-Keller-N3LO", "ceft-Keller-N3LO-MOD", "BPS" or None if no crust is needed')

        self.crust = crust
        self.rho_t = rho_t
        if crust is not None:
            self.BPS = self.get_BPS()
            self.ceft = crust[0:4] == 'ceft'

            if self.ceft is True:

                if crust == 'ceft-Hebeler':
                    self.min_norm = 1.676
                    self.max_norm = 2.814
                    self.min_index = 2.486
                    self.max_index = 2.571
                    self._rho_start_ceft = 0.5792
                    self._rho_end_BPS = 0.5

                if crust == 'ceft-Drischler':
                    self.min_norm = 2.136
                    self.max_norm = 3.339
                    self.min_index = 2.623
                    self.max_index = 2.951
                    self._rho_start_ceft = 0.5625
                    self._rho_end_BPS = 0.5

                if crust == 'ceft-Lynn':
                    self.min_norm = 0.975
                    self.max_norm = 2.784
                    self.min_index = 1.844
                    self.max_index = 2.628
                    self._rho_start_ceft = 0.625
                    self._rho_end_BPS = 0.5

                if crust == 'ceft-Tews':
                    self.min_norm = 2.155
                    self.max_norm = 3.176
                    self.min_index = 2.991
                    self.max_index = 2.586
                    self._rho_start_ceft = 0.5792
                    self._rho_end_BPS = 0.5

                if crust == 'ceft-Keller-LO':
                    self.min_norm = 1.624
                    self.max_norm = 1.624000000001  #to mimic the existence of a max band
                    self.min_index = 1.517
                    self.max_index = 1.517
                    self._rho_start_ceft = 0.5792
                    self._rho_end_BPS = 0.5

                #if crust == 'ceft-Keller-NLO-0.5':
                #    self.min_norm = 0.548
                #    self.max_norm = 1.286
                #    self.min_index = 2.110
                #    self.max_index = 1.729
                #    self._rho_start_ceft = 0.5792
                #    self._rho_end_BPS = 0.225

                if crust == 'ceft-Keller-NLO':
                    self.min_norm = 0.531
                    self.max_norm = 1.283
                    self.min_index = 2.229
                    self.max_index = 1.738
                    self._rho_start_ceft = 0.7
                    self._rho_end_BPS = 0.35

                if crust == 'ceft-Keller-N2LO':
                    self.min_norm = 1.81356
                    self.max_norm = 3.49759
                    self.min_index = 2.39119
                    self.max_index = 3.00198
                    self._rho_start_ceft = 0.5792
                    self._rho_end_BPS = 0.5

                if crust == 'ceft-Keller-N3LO':
                    self.min_norm = 2.207
                    self.max_norm = 3.056
                    self.min_index = 2.361
                    self.max_index = 2.814
                    self._rho_start_ceft = 0.5792
                    self._rho_end_BPS = 0.5

                #if crust == 'ceft-Keller-N3LO-MOD':    # to mimic a distribution in ceft norm
                #    self.min_norm = 2.207
                #    self.max_norm = 3.056
                #    self.min_index = 2.814
                #    self.max_index = 2.814
                #    self._rho_start_ceft = 0.5792
                #    self._rho_end_BPS = 0.5

                if crust == 'ceft-old':
                    self.min_norm = 1.7
                    self.max_norm = 2.76
                    self.min_index = 2.5
                    self.max_index = 2.5
                    self._rho_start_ceft = 0.5792
                    self._rho_end_BPS = 0.5

            if rho_t > 2.0 * rho_ns or rho_t < self._rho_start_ceft * rho_ns:
                raise ValueError('The transition density should be between \
                    %.2f and 2.0 saturation density.' % self._rho_start_ceft)

    def update(self, eos_params, max_edsc=True):
        """
        Method to update a given EoS object with specified parameters.


        Parameters
        ----------
        eos_params : dict
            A dictionary containing the parameter values of the EoS model.
        max_edsc: bool
            If True, compute the maximum central energy density allowed
            by this set of parameters (default is True).

        """

        if self.crust is not None:

            if self.ceft is True:
                self.ceft_param = eos_params['ceft']
                self.ceft_param_in = eos_params['ceft_in']                   #### added this
                self.eos_params = {i:eos_params[i] for i in eos_params if 
                                   i != 'ceft' or i != 'ceft_in'}

                if (self.ceft_param < self.min_norm or
                        self.ceft_param > self.max_norm):
                    raise TypeError('"ceft" variable should be either "None" \
                                    or a float in the range [1.7, 2.76]')
                self.get_eos_crust()

            else:
                self.eos_params = {i:eos_params[i] for i in eos_params}
                self.get_eos_crust()

        else:
            self.eos_params = {i:eos_params[i] for i in eos_params}
        
        self.get_eos()

        if max_edsc is True:
            self.find_max_edsc()
        else:
            self.max_edsc = 0.0

    # Compute the crust EoS
    def get_eos_crust(self):
        if self.ceft is True:
            # TODO: add function that rho_t can be below 0.58*rho_ns
            # attempt at making a different jump off from BPS  ###DONE
            
            rhocrust = self.BPS[:,0][self.BPS[:,0] <= self._rho_end_BPS]
            rhotrans = numpy.linspace(self._rho_end_BPS, self._rho_start_ceft, 10)  #increased number of points here from 10 to 20/50, didn't work
            rhocEFT = numpy.linspace(self._rho_start_ceft, 
                                     self.rho_t / rho_ns, 10)  # from 10 to 20/50, didn't work

            prescrust = self.BPS[:,1][self.BPS[:,0] <= self._rho_end_BPS]
            #prescEFT = self.ceft_band_func(rhocEFT, self.ceft_param,
            #                               self.min_norm, self.max_norm,
            #                               self.min_index, self.max_index)
            
            #index is determined already, we get it from there
            prescEFT = self.polytropic_func(rhocEFT, self.ceft_param, self.ceft_param_in)   ####### added this line

            prestrans = prescrust[-1] * (rhotrans / rhocrust[-1])**(
                numpy.log10(prescEFT[0] / prescrust[-1]) /
                numpy.log10(rhocEFT[0] / rhocrust[-1]))


            rholow = numpy.logspace(-2, numpy.log10(self.BPS[0][0] *
                                                    rho_ns), 50)
            epslow = numpy.logspace(-2, numpy.log10(self.BPS[0][2] /
                                                    gcm3_to_MeVfm3), 50)
            preslow = ((epslow / (self.BPS[0][0] * rho_ns))**(5. / 3.) *
                       self.BPS[0][1] / dyncm2_to_MeVfm3)

            self._rho_crust = numpy.hstack([rhocrust, rhotrans[1:-1], 
                                            rhocEFT]) * rho_ns
            self._pres_crust = numpy.hstack([prescrust, prestrans[1:-1], 
                                             prescEFT]) / dyncm2_to_MeVfm3

            eps0 = self.BPS[:,2][self.BPS[:,0] == self._rho_end_BPS] / gcm3_to_MeVfm3
            prho = UnivariateSpline(self._rho_crust, 
                                    self._pres_crust, k=2, s=0)

            result = odeint(self.edens, eps0, 
                            self._rho_crust[self._rho_crust / rho_ns >= self._rho_end_BPS],
                            args=tuple([prho]))
            eds_crust = result.flatten()[1::]

            self._eds_crust = numpy.hstack([epslow[0:-1], 
                                            self.BPS[:,2][self.BPS[:,0] <= self._rho_end_BPS]
                                            / gcm3_to_MeVfm3, eds_crust])
            self._pres_crust = numpy.hstack([preslow[0:-1], self._pres_crust])
            self._rho_crust = numpy.hstack([rholow[0:-1], self._rho_crust])

        if self.ceft is False:

            rhocrust = self.BPS[:,0][self.BPS[:,0] <= self.rho_t / rho_ns]
            prescrust = self.BPS[:,1][self.BPS[:,0] <= self.rho_t / rho_ns]
            edscrust = self.BPS[:,2][self.BPS[:,0] <= self.rho_t / rho_ns]

            rholow = numpy.logspace(-2, numpy.log10(self.BPS[0][0] * 
                                                    rho_ns), 50)
            epslow = numpy.logspace(-2, numpy.log10(self.BPS[0][2] /
                                                    gcm3_to_MeVfm3), 50)
            preslow = ((epslow / (self.BPS[0][0] * rho_ns))**(5. / 3.) *
                       self.BPS[0][1])

            self._rho_crust = rhocrust * rho_ns
            self._pres_crust = prescrust / dyncm2_to_MeVfm3

            self._eds_crust = numpy.hstack([epslow[0:-1], 
                                            edscrust / gcm3_to_MeVfm3])
            self._pres_crust = numpy.hstack([preslow[0:-1], self._pres_crust])
            self._rho_crust = numpy.hstack([rholow[0:-1], self._rho_crust])

        eos_crust = UnivariateSpline(self._eds_crust, 
                                     self._pres_crust, k=1, s=0)
        self._cs_crust = eos_crust.derivative(1)
        self.rhoeds_crust = UnivariateSpline(self._rho_crust, 
                                             self._eds_crust, k=1, s=0)
        self.eds_t = self._eds_crust[-1]
        self.P_t = self._pres_crust[-1]

    #######################
    # Auxiliary functions #
    #######################

    # Analytic representation of the SLy EoS, used for crust
    def SLYfit(self, rho):

        a = numpy.array([6.22, 6.121, 0.005925, 0.16326, 6.48, 11.4971, 19.105,
                         0.8938, 6.54, 11.4950, -22.775, 1.5707, 4.3, 14.08,
                         27.80, -1.653, 1.50, 14.67])
        part1 = ((a[0] + a[1] * rho + a[2] * rho**3.) / (1. + a[3] * rho) *
                 1. / (numpy.exp(a[4] * (rho - a[5])) + 1.))
        part2 = ((a[6] + a[7] * rho) * 1. / 
                 (numpy.exp(a[8] * (a[9] - rho)) + 1.))
        part3 = ((a[10] + a[11] * rho) * 1. /
                 (numpy.exp(a[12] * (a[13] - rho)) + 1.))
        part4 = ((a[14] + a[15] * rho) * 1. /
                 (numpy.exp(a[16] * (a[17] - rho)) + 1.))
        pres = part1 + part2 + part3 + part4
        return pres

    def edens(self, eps, rho, eos):
        dedrho = (eos(rho) / c**2. + eps) / rho  #eos(rho) is a pressure, which when divided by c^2 gives units of g/cm^3
        return dedrho

    def rhodens(self, rho, eps):
        eos = UnivariateSpline(self._eds_core, self._pres_core, k=1, s=0)
        drhode = rho / (eos(eps) / c**2. + eps) #same reasoning as above
        return drhode

    def PofE(self,E, epsgrid, presgrid):
        idx = numpy.searchsorted(epsgrid, E)
        if idx == 0:
            pres = presgrid[0] * pow(E / epsgrid[0], 5. / 3.)
        if idx == len(epsgrid): 
            pres = presgrid[-1] * pow(E / epsgrid[-1], 5. / 3.)
        else:
            ci = numpy.log(presgrid[idx] / presgrid[idx - 1]) / numpy.log(epsgrid[idx] / epsgrid[idx - 1])
            pres = presgrid[idx - 1] * (E / epsgrid[idx - 1])**ci
        return pres

    def CofE(self, E, epsgrid, presgrid):
        idx = numpy.searchsorted(epsgrid, E)
        ci = (numpy.log(presgrid[idx] / presgrid[idx - 1]) /
              numpy.log(epsgrid[idx] / epsgrid[idx - 1]))
        return ((presgrid[idx - 1] * epsgrid[idx - 1]**(-1) * ci *
                (E / epsgrid[idx - 1])**(ci - 1.)) / c**2.)

    def EofP(self, P, epsgrid, presgrid):
        idx = numpy.searchsorted(presgrid, P)
        if idx == 0:
            eds = epsgrid[0] * pow(P / presgrid[0], 3. / 5.)
        if idx == len(epsgrid): 
            eds = epsgrid[-1] * pow(P / presgrid[-1], 3. / 5.)
        else:
            ci = (numpy.log(presgrid[idx] / presgrid[idx - 1]) /
                  numpy.log(epsgrid[idx] / epsgrid[idx - 1]))
            eds = epsgrid[idx - 1] * pow(P / presgrid[idx - 1], 1. / ci)
        return eds

    def EofRho(self, rho, rhos, eps):
        idx = numpy.searchsorted(rhos, rho)
        ci = (numpy.log(eps[idx] / eps[idx - 1]) /
              numpy.log(rhos[idx] / rhos[idx - 1]))
        return eps[idx - 1] * (rho / rhos[idx - 1])**ci


    # Find maximum central energy density
    def find_max_edsc(self):

        min_edsc0 = 14.2         #changed from 14.2 to 8 for NLO (9/10 didn't work)
        if self.rho_t is not None:
            eds = numpy.logspace(numpy.log10(self.rho_t), 
                                 numpy.log10(4e16), 1000) #eds is in units of g/cm^3,  
                                                                
        else:
            eds = numpy.logspace(14.2, numpy.log10(4e16), 1000) #same as above        #changed from 14.2 to 8 for NLO (9/10 didn't work)
        dpde = self.eos.derivative(1)
        cs = dpde(eds)/c**2
        acausal = 1.

        if len(eds[cs > acausal]) != 0:
            maximum = eds[numpy.where(eds == min(eds[cs > acausal]))[0] - 1]

            if numpy.log10(maximum) < min_edsc0: # g/cm^3
                maximum = min_edsc0 + 0.01

        else:
            maximum = max(eds)
 
        eds_c = numpy.logspace(min_edsc0, numpy.log10(maximum), 50) # g/cm^3
        Ms = numpy.zeros((len(eds_c),3))
        
        for i, e in enumerate(eds_c):
            star = Star(e)
            star.solve_structure(self.energydensities, self.pressures)
            if star.Mrot < Ms[i - 1][0]:
                break
            Ms[i] = star.Mrot, star.Req, star.tidal

        Ms = Ms[Ms[:,0] > 0.0]
        eds_c = eds_c[0:len(Ms)]
        test, idx = numpy.unique(Ms[:,0], return_index=True)
        Ms = Ms[idx]

        eds_c = eds_c[idx]
 
        self.max_M = max(Ms[:,0])
        index_max_M = numpy.argmax(Ms[:,0])
        self.Radius_max_M = Ms[:,1][index_max_M]
        self.max_edsc = max(eds_c)
        self.min_edsc = 10**(min_edsc0)
        if Ms[:,0][-1] > 0.9:
            self.min_edsc = min(eds_c[Ms[:,0] > 0.9])
        self.centraleds = eds_c
        self.massradius = Ms


    def get_minmax_edsc_chirp(self, chirp):

        spline = UnivariateSpline(self.massradius[:,0], self.centraleds,
                                  k=1, s=0)
        min_mass = m1_from_mc_m2(chirp, self.max_M)
        max_mass = max(m1_m2_from_mc_q(chirp, 1.))
        if self.massradius[:,0][0] < min_mass < self.massradius[:,0][-1]:
            min_edsc = spline(min_mass)
        else:
            min_edsc = self.min_edsc
        if self.massradius[:,0][0] < max_mass < self.massradius[:,0][-1]:
            max_edsc = spline(max_mass)
        else:
            max_edsc = self.max_edsc

        return max(min_edsc, self.min_edsc), min(max_edsc, self.max_edsc)

    def polytropic_func(self, rho, K, index):
        return K * rho**index

    #def ceft_band_func(self, rho, norm, min_norm,                 ### we don't use it anymore
    #                   max_norm, min_index, max_index):

    #    index = ((norm - min_norm) / (max_norm - min_norm) *
    #             (max_index - min_index) + min_index)
    #    return self.polytropic_func(rho, norm, index)


    # Plotting functions #
    def plot(self):
        """
            Plot the EoS.
        """

        fig, ax = pyplot.subplots(1,1, figsize=(8,6))
        if self.crust is not None:
            rho_crust = numpy.logspace(13., numpy.log10(self.rho_t),
                                       100)  #rho_crust is in units of g/cm^3 
                                                        
            rho_core = numpy.logspace(numpy.log10(self.rho_t), 
                                      numpy.log10(8e15), 100) #same as above, but for rho_core

            miny = min(self.eos(rho_crust)) #   units of g/(cm s^2)
                                                        
            maxy = max(self.eos(rho_core)) #same deal as above

            ax.plot(rho_crust, self.eos(rho_crust), 
                    c='red', label='Crust EoS', lw=1.5) 
            ax.plot(rho_core, self.eos(rho_core),
                    c='black', label='Core EoS', lw=1.5) 

        else:
            miny = min(self.eos(self.energydensities)) #units of g/(cm s^2)
            maxy = max(self.eos(self.energydensities)) #same as above
            ax.plot(self.energydensities, 
                    self.pressures, c='black', lw=1.5, label='EoS') #same as above, but for energydensities and pressures

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(miny, maxy)


        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel(r'$\varepsilon$ [g/cm$^3$]', fontsize=15)
        ax.set_ylabel(r'Pressure [dyn/cm$^2$]', fontsize=15)
        ax.legend(prop={'size': 12})
        pyplot.tight_layout()
        fig.savefig('testEoS_cgs.png')
        pyplot.show()

    def plot_massradius(self):
        """
            Plot the mass-radius curve.
        """
        fig, ax = pyplot.subplots(1,1, figsize=(8,6))

        ax.scatter(self.massradius[:,1], self.massradius[:,0], c='black')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel(r'Radius [km]', fontsize=15)
        ax.set_ylabel(r'Mass [M$_{\odot}$]', fontsize=15)
        ax.set_xlim(8, 16)
        ax.set_ylim(0.2, 3.)
        pyplot.tight_layout()
        fig.savefig('testEoS_MR_cgs.png')
        pyplot.show()

    def get_BPS(self):
        BPS = numpy.array([[2.97500000e-14, 6.30300000e-24, 4.40700000e-12], [3.06900000e-14, 6.30300000e-23, 4.54700000e-12], [4.36900000e-14, 7.55100000e-22, 6.47200000e-12],
                           [6.18700000e-14, 8.73700000e-21, 9.15000000e-12], [1.70000000e-13, 1.06100000e-19, 2.51600000e-11], [7.93700000e-13, 3.63200000e-18, 1.18300000e-10],
                           [4.33100000e-12, 1.18600000e-16, 6.41600000e-10], [3.93400000e-11, 6.08100000e-15, 5.82500000e-09], [9.88100000e-11, 3.10000000e-14, 1.46300000e-08],
                           [2.48300000e-10, 1.51700000e-13, 3.67500000e-08], [6.23500000e-10, 7.18300000e-13, 9.22800000e-08], [1.56600000e-09, 3.28600000e-12, 2.31900000e-07],
                           [3.93400000e-09, 1.44700000e-11, 5.82500000e-07], [9.88100000e-09, 6.08800000e-11, 1.46300000e-06], [2.48300000e-08, 2.44100000e-10, 3.67600000e-06],
                           [3.12500000e-08, 3.28200000e-10, 4.62700000e-06], [6.23500000e-08, 8.95500000e-10, 9.23300000e-06], [1.24400000e-07, 2.39200000e-09, 1.84200000e-05],
                           [2.48200000e-07, 6.27800000e-09, 3.67600000e-05], [4.95300000e-07, 1.62500000e-08, 7.33700000e-05], [9.88100000e-07, 4.16600000e-08, 1.46400000e-04],
                           [1.24400000e-06, 5.45300000e-08, 1.84300000e-04], [1.97200000e-06, 1.01700000e-07, 2.92200000e-04], [3.12500000e-06, 1.89000000e-07, 4.63100000e-04],
                           [3.93400000e-06, 2.57700000e-07, 5.83000000e-04], [4.95200000e-06, 3.14300000e-07, 7.34200000e-04], [6.23500000e-06, 4.28100000e-07, 9.24500000e-04],
                           [9.88100000e-06, 7.93800000e-07, 1.46500000e-03], [1.56600000e-05, 1.47000000e-06, 2.32300000e-03], [2.48200000e-05, 2.72200000e-06, 3.68300000e-03],
                           [3.12500000e-05, 3.53300000e-06, 4.63700000e-03], [3.93400000e-05, 4.80700000e-06, 5.83600000e-03], [4.95200000e-05, 6.54000000e-06, 7.35300000e-03],
                           [6.23500000e-05, 8.89300000e-06, 9.25600000e-03], [7.85000000e-05, 1.20900000e-05, 1.16600000e-02], [9.88100000e-05, 1.56200000e-05, 1.46800000e-02],
                           [1.24400000e-04, 2.12400000e-05, 1.84800000e-02], [1.56600000e-04, 2.88800000e-05, 2.32800000e-02], [1.97200000e-04, 3.71300000e-05, 2.93100000e-02],
                           [2.48200000e-04, 5.04800000e-05, 3.69200000e-02], [3.12500000e-04, 6.86500000e-05, 4.64900000e-02], [3.93400000e-04, 9.33000000e-05, 5.85200000e-02],
                           [4.95300000e-04, 1.26900000e-04, 7.37600000e-02], [6.23500000e-04, 1.62100000e-04, 9.28400000e-02], [6.90600000e-04, 1.80500000e-04, 1.02900000e-01],
                           [7.85000000e-04, 2.05300000e-04, 1.16900000e-01], [9.88100000e-04, 2.79100000e-04, 1.47300000e-01], [1.24400000e-03, 3.63000000e-04, 1.85500000e-01],
                           [1.60800000e-03, 4.87100000e-04, 2.39800000e-01], [1.95400000e-03, 5.21200000e-04, 2.91700000e-01], [2.46900000e-03, 5.67800000e-04, 3.68800000e-01],
                           [2.97400000e-03, 6.13500000e-04, 4.44300000e-01], [3.63300000e-03, 6.75900000e-04, 5.42700000e-01], [4.46400000e-03, 7.60100000e-04, 6.67300000e-01],
                           [5.49100000e-03, 8.73100000e-04, 8.20700000e-01], [2.50000000e-02, 5.79900000e-03, 3.76200000e+00], [5.00000000e-02, 1.16600000e-02, 7.53000000e+00],
                           [7.50000000e-02, 2.08500000e-02, 1.13000000e+01], [1.00000000e-01, 3.21600000e-02, 1.50800000e+01], [1.25000000e-01, 4.51500000e-02, 1.88600000e+01],
                           [1.50000000e-01, 5.96100000e-02, 2.26400000e+01], [1.75000000e-01, 7.54400000e-02, 2.64200000e+01], [2.00000000e-01, 9.26000000e-02, 3.02100000e+01],
                           [2.25000000e-01, 1.11000000e-01, 3.40000000e+01], [2.50000000e-01, 1.30800000e-01, 3.77900000e+01], [2.75000000e-01, 1.51800000e-01, 4.15800000e+01],
                           [3.00000000e-01, 1.74200000e-01, 4.53800000e+01], [3.25000000e-01, 1.98000000e-01, 4.91800000e+01], [3.50000000e-01, 2.23100000e-01, 5.29800000e+01],
                           [3.75000000e-01, 2.49700000e-01, 5.67800000e+01], [4.00000000e-01, 2.77700000e-01, 6.05800000e+01], [4.25000000e-01, 3.07300000e-01, 6.43800000e+01],
                           [4.50000000e-01, 3.38500000e-01, 6.81900000e+01], [4.75000000e-01, 3.71300000e-01, 7.20000000e+01], [5.00000000e-01, 4.05400000e-01, 7.58100000e+01],
                           [5.79200000e-01, 4.34030555e-01, 8.78844243e+01], [6.37066667e-01, 5.50694991e-01, 9.67134788e+01]])
        return BPS
