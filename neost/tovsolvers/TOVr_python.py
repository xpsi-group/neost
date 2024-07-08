import numpy as np
from numba import jit, double
import numba
from scipy.integrate import solve_ivp

from .. import global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s

@jit(nopython=True)
def pressure_epsilon(P, epsgrid, presgrid):
    idx = np.searchsorted(presgrid, P)
    if idx == 0:
        eds = epsgrid[0] * np.power(P / presgrid[0], 3. / 5.)
    if idx == len(presgrid):
        eds = epsgrid[-1] * np.power(P / presgrid[-1], 3. / 5.)
    else:
        ci = np.log(presgrid[idx] / presgrid[idx - 1]) / np.log(epsgrid[idx] / epsgrid[idx - 1])
        eds = epsgrid[idx - 1] * np.power(P / presgrid[idx - 1], 1. / ci)
    return eds

@jit(nopython=True)
def epsilon_pressure(E, epsgrid, presgrid):
    idx = np.searchsorted(epsgrid, E)
    if idx == 0:
        pres = presgrid[0] * np.power(E / epsgrid[0], 5. / 3.)
    if idx == len(epsgrid): 
        pres = presgrid[-1] * np.power(E / epsgrid[-1], 5. / 3.)
    else:
        ci = np.log(presgrid[idx] / presgrid[idx - 1]) / np.log(epsgrid[idx] / epsgrid[idx - 1])
        pres = presgrid[idx - 1] * (E / epsgrid[idx - 1])**ci
    return pres

@jit(nopython=True)
def pressure_adind(P, epsgrid, presgrid):
    idx = np.searchsorted(presgrid, P)
    if idx == 0:
        eds = epsgrid[0] * np.power(P / presgrid[0], 3. / 5.)
        adind = 5. / 3. * presgrid[0] * np.power(eds / epsgrid[0], 5. / 3.) * 1. / eds * (eds + P) / P
    if idx == len(presgrid):
        eds = epsgrid[-1] * np.power(P / presgrid[-1], 3. / 5.)
        adind = 5. / 3. * presgrid[-1] * np.power(eds / epsgrid[-1], 5. / 3.) * 1. / eds * (eds + P) / P
    else:
        ci = np.log(presgrid[idx] / presgrid[idx-1]) / np.log(epsgrid[idx] / epsgrid[idx-1])
        eds = epsgrid[idx-1] * np.power(P / presgrid[idx-1], 1. / ci)
        adind = ci * presgrid[idx-1] * np.power(eds / epsgrid[idx-1], ci) * 1. / eds * (eds + P) / P
    return adind

@jit(nopython=True)
def TOV(r, y, epsgrid, presgrid):

    p = y[0]
    eps = pressure_epsilon(p, epsgrid, presgrid)
    ad_index = pressure_adind(p, epsgrid, presgrid)

    dpdr = -(eps + p) * (y[1] + 4. * np.pi * np.power(r,3) * p)
    dpdr *= np.power(r * (r - 2. * y[1]), -1)
    dmdr = 4. * np.pi * np.power(r,2) * eps

    dhdr = y[3]
    dfdr = 2. * np.power(1. - 2. * y[1] / r, -1) * y[2] * \
        (-2. * np.pi * (5. * eps + 9. * p + (eps + p)**2. /
                        (p * ad_index)) + 3. / np.power(r,2) + 2. *
            np.power(1. - 2. * y[1] / r,-1) * np.power(y[1] / np.power(r,2) +
         4. * np.pi * r * p,2)) \
        + 2. * y[3] / r * np.power(1. - 2. * y[1] / r, -1) * \
        (-1. + y[1] / r + 2. * np.pi * np.power(r,2) * (eps - p))

    dalphadr = (y[1] + 4. * np.pi * np.power(r,3) * p) *\
        np.power(r * (r - 2. * y[1]), -1)


    return np.array([dpdr, dmdr, dhdr, dfdr, dalphadr])

@jit(nopython=True)
def Q22(x):
    q22 = 3. / 2. * (np.power(x,2) - 1.) * np.log((x + 1.) / (x - 1.)) -\
        (3. * np.power(x,3) - 5. * x) / (np.power(x,2) - 1.)
    return q22

@jit(nopython=True)
def Q21(x):
    q21 = np.sqrt(np.power(x,2) - 1.) *\
        ((3. * np.power(x,2) - 2.) /\
         (np.power(x,2) - 1.) - 3. * x /\
         2. * np.log((x + 1.) / (x - 1.)))
    return q21


def initial_conditions(epscent, pcent, adindcent=2.):
        """
        Set the initial conditions for solving the structure equations. 

        Args: 
            eos (object): An object that takes energy density as input and outputs pressure, both in cgs units.
            w0 (float): The initial value of the rotational drag. Not known a priori, but can be calculated after the TOV equations are solved.
            j0 (float): The initial value of j. Not known a priori, but can be calculated after the TOV equations are solved.
            static (bool): Calculate initial conditions for a static star (True) or a rotating star (False). 

        Returns:
            tuple: tuple containing:

                - **dr** (*float*): The initial stepsize in cm. 
                - **intial**   (*array*): A np array storing the initial conditions.

        """
        if hasattr(pcent, '__len__') or hasattr(epscent, '__len__'):
            # pcent and epscent are sometimes scalars and sometimes arrays of length 1,
            # causing ragged arrays which numpy no longer accepts.
            # Therefore, check if:
            # 1. Either of them are arrays
            # 2. If so, that both of them are
            # 3. That they have the same shape
            # 4. And that that shape is (1,).
            # If this is true, convert them to scalars to avoid ragged arrays.
            try:
                assert(hasattr(pcent, '__len__') and hasattr(epscent, '__len__'))
                assert(pcent.shape == epscent.shape)
                assert(pcent.shape == (1,))
            except AssertionError:
                raise ValueError('The python TOV solver has tried to create a ragged numpy array. This is no longer supported.')
            pcent = pcent[0]
            epscent = epscent[0]

        r = 4.441e-16
        dr = 10.

        P0 = pcent - (2. * np.pi / 3.) * (pcent + epscent) * \
            (3. * pcent + epscent) * r**2.
        m0 = 4. / 3. * np.pi * epscent * r**3.
        h0 = r**2.
        b0 = 2. * r

        initial = np.array([P0, m0, h0, b0, 0.0])

        return dr, initial

@jit(nopython=True)
def tidal_deformability(y2, Mns, Rns):

    C = Mns / Rns
    Eps = 4. * C**3. * (13. - 11. * y2 + C * (3. * y2 - 2.) +
                        2. * C**2. * (1. + y2)) + \
        3. * (1. - 2. * C)**2. * (2. - y2 + 2. * C * (y2 - 1.)) * \
        np.log(1. - 2. * C) + 2. * C * (6. - 3. * y2 + 3. * C * (5. * y2 - 8.))
    tidal_def = 16. / (15. * Eps) * (1. - 2. * C)**2. *\
        (2. + 2. * C * (y2 - 1.) - y2)

    return tidal_def


def solveTOVr(epscent, eos_eps, eos_pres, atol, rtol, hmax, step): #assumed to be in cgs units as inputs eps has units of g/cm^3 and pres has units g/(cm s^2)

    eos_pres, indices = np.unique(np.log10(eos_pres).round(decimals=3), return_index=True)
    eos_pres = 10**eos_pres * G * np.power(c,-4) #scaled into geometrized
    eos_eps = eos_eps[np.sort(indices)] * G * np.power(c,-2) #scaled into geometrized

    # Set initial conditions for solving the TOV equations ##
    epscent = epscent * G * np.power(c,-2)
    pcent = epsilon_pressure(epscent, eos_eps, eos_pres)
    adindcent = pressure_adind(pcent, eos_eps, eos_pres)

    r = 4.441e-16
    rmax = 50 * 1e5
    dr, initial = initial_conditions(epscent, pcent, adindcent)

    # Integrate the TOV equations
    sol_in = solve_ivp(TOV, t_span=(r, rmax), y0=initial,
                       method='DOP853', t_eval=None, args=(eos_eps, eos_pres),
                       max_step=5000.)

    Mb = sol_in.y[:,-1][1] # in geometrized units
    Rns = sol_in.t[-1] #in units of cm
    y = Rns * sol_in.y[:,-1][3] / sol_in.y[:,-1][2]
    tidal = tidal_deformability(y, Mb, Rns)

    Gtt = np.zeros((len(sol_in.t), 2))
    Gtt[:,0] = sol_in.t #radius in cm
    Gtt[:,1] = sol_in.y[4] - (sol_in.y[:,-1][4] - 0.5 * np.log(1 - 2 * Mb / Rns))

    Mb = Mb * np.power(c,2) / G #now in units of grams

    return Mb, Rns, tidal, Gtt
