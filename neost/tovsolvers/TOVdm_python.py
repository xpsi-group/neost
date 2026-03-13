from numba import jit, float64
import numpy as np
from math import pow, log
from scipy.integrate import solve_ivp

from .. import global_imports

dyncm2_to_MeVfm3 = global_imports._dyncm2_to_MeVfm3
gcm3_to_MeVfm3 = global_imports._gcm3_to_MeVfm3
oneoverfm_MeV = global_imports._oneoverfm_MeV
c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s


@jit(nopython=True)
def TOV_complete(r, Z, epsgrid_dm, presgrid_dm, epsgrid, presgrid):
    """
    Calculate the derivatives of the mass and pressure for both the baryonic and dark matter components, as well as the metric function alpha, at a given radius `r` and state vector `Z`. This function is used to solve the complete TOV equations for a neutron star with a dark matter component. The state vector `Z` contains the following components:
    - Z[0]: Mass of the baryonic component enclosed within radius `r` (mb)
    - Z[1]: Mass of the dark matter component enclosed within radius `r` (mchi)
    - Z[2]: Pressure of the baryonic component at radius `r` (pb)
    - Z[3]: Pressure of the dark matter component at radius `r` (pchi)
    - Z[4]: Metric function alpha(r) related to the time component of the metric

    Args:
        r (float): Radial coordinate at which to evaluate the derivatives.
        Z (np.ndarray): The state vector at radius `r`, containing the components described above.
        epsgrid_dm (np.ndarray): Grid of energy density values for the dark matter component in geometrized units (g/cm^3 converted to g/cm).
        presgrid_dm (np.ndarray): Grid of pressure values for the dark matter component in geometrized units (g/(cm s^2) converted to g/(cm s^2)).
        epsgrid (np.ndarray): Grid of energy density values for the baryonic component in geometrized units (g/cm^3 converted to g/cm).
        presgrid (np.ndarray): Grid of pressure values for the baryonic component in geometrized units (g/(cm s^2) converted to g/(cm s^2)).

    Returns:
        np.ndarray: An array containing the derivatives of the mass and pressure for both components, as well as the derivative of the metric function alpha, in the following order:
        - dmbdr: Derivative of the mass of the baryonic component with respect to radius (dmb/dr)
        - dmchidr: Derivative of the mass of the dark matter component with respect to radius (dmchi/dr)
        - dpbdr: Derivative of the pressure of the baryonic component with respect to radius (dpb/dr)
        - dpchidr: Derivative of the pressure of the dark matter component with respect to radius (dpchi/dr)
        - dalphadr: Derivative of the metric function alpha with respect to radius (dalpha/dr)
    """
    mb = Z[0]
    mchi = Z[1]
    M = mb + mchi
    pb = Z[2]
    pchi = Z[3]
    P = pb + pchi
    dalphadr = (M + 4 * np.pi * r**3. * P) / (r**2. - 2 * M * r)
    eb = EofP(pb, epsgrid, presgrid)
    echi = EofP(pchi, epsgrid_dm, presgrid_dm)
    dmbdr = 4 * np.pi * r**2. * eb 
    dmchidr = 4 * np.pi * r**2. * echi 
    dpbdr = -(eb + pb) * dalphadr
    dpchidr = -(echi + pchi) * dalphadr
    return np.array([dmbdr,dmchidr,dpbdr, dpchidr, dalphadr])


@jit(nopython=True)
def TOV_single(r, Z, epsgrid, presgrid):
    """
    Calculate the derivatives for the TOV equations. This function is used to solve the single-fluid TOV equations for either the baryonic or dark matter component after the complete TOV equations have been solved and one of the components has dropped to zero pressure. The state vector `Z` contains the following components:
    - Z[0]: Mass of the component enclosed within radius `r`
    - Z[1]: Pressure of the component at radius `r`

    Args:
        r (float): Radial coordinate at which to evaluate the derivatives.
        Z (np.ndarray): The state vector at radius `r`, containing the components described above   (mass and pressure of the remaining component). 
        epsgrid (np.ndarray): Grid of energy density values for the component in geometrized units (g/cm^3 converted to g/cm).
        presgrid (np.ndarray): Grid of pressure values for the component in geometrized units (g/(cm s^2) converted to g/(cm s^2)).
    Returns:
        np.ndarray: An array containing the derivatives of the mass and pressure for the remaining component, as well as the derivative of the metric function alpha, in the following order:
        - dmbdr: Derivative of the mass of the component with respect to radius (dm/dr)
        - dpbdr: Derivative of the pressure of the component with respect to radius (dp/dr) 
        - dalphadr: Derivative of the metric function alpha with respect to radius (dalpha/dr)
    """
    mb = Z[0]
    pb = Z[1]
    P = pb
    M = mb
    dalphadr = (M + 4 * np.pi * r**3. * P) /(r**2. - 2 * M * r)
    eb = EofP(pb, epsgrid, presgrid)
    dmbdr = 4 * np.pi * r**2. * eb
    dpbdr = -(eb + pb) * dalphadr
    return np.array([dmbdr,dpbdr, dalphadr])


@jit(float64(float64, float64[:], float64[:]), nopython=True)
def EofP(P, epsgrid, presgrid):
    idx = np.searchsorted(presgrid, P)
    if idx == 0:
        eds = epsgrid[0] * pow(P / presgrid[0], 3. / 5.)
    if idx == len(presgrid):
        eds = epsgrid[-1] * pow(P / presgrid[-1], 3. / 5.)
    else:
        ci = np.log(presgrid[idx] / presgrid[idx-1]) /\
            np.log(epsgrid[idx] / epsgrid[idx-1])
        eds = epsgrid[idx-1] * pow(P / presgrid[idx-1], 1. / ci)
    return eds


@jit(float64(float64, float64[:], float64[:]), nopython=True)
def PofE(E, epsgrid, presgrid):
    idx = np.searchsorted(epsgrid, E)
    if idx == 0:
        pres = presgrid[0] * pow(E / epsgrid[0], 5. / 3.)
    if idx == len(epsgrid):
        pres = presgrid[-1] * pow(E / epsgrid[-1], 5. / 3.)
    else:
        ci = np.log(presgrid[idx] / presgrid[idx - 1]) / np.log(epsgrid[idx] / epsgrid[idx - 1])
        pres = presgrid[idx - 1] * (E / epsgrid[idx - 1])**ci
    return pres


def solveTOVdm(epscent, epscent_dm, eps, pres, eps_dm, pres_dm, dm_halo, two_fluid_tidal, atol, rtol, hmax, step):

    """Solve the TOV equations for a neutron star with a dark matter component, which can be either a core or a halo. The TOV equations are solved in two steps: first, we solve the complete 
    TOV equations for both the baryonic and dark matter components until the pressure of either component drops to
     zero. Then, we solve the single-fluid TOV equations for the remaining component until its pressure drops to
    zero. The function returns the mass and radius of the neutron star, as well as the mass and radius of the 
    dark matter core and halo (if present), and the tidal deformability if requested.
    
    Args:        epscent (float): The central energy density of the baryonic component in cgs units (g/cm^3).
        epscent_dm (float): The central energy density of the dark matter component in cgs units (g/cm^3).
        eps (np.ndarray): Grid of energy density values for the baryonic component in cgs units (g/cm^3).
        pres (np.ndarray): Grid of pressure values for the baryonic component in cgs units (g/(cm s^2)).
        eps_dm (np.ndarray): Grid of energy density values for the dark matter component in cgs units (g/cm^3).
        pres_dm (np.ndarray): Grid of pressure values for the dark matter component in cgs units (g/(cm s^2)).
        dm_halo (bool): Whether to solve for a dark matter halo (True) or just a dark matter core (False).
        two_fluid_tidal (bool): Whether to calculate the tidal deformability using the two-fluid TOV equations (True) or just the single-fluid TOV equations (False).
        atol (float): Absolute tolerance for the ODE solver.
        rtol (float): Relative tolerance for the ODE solver.
        hmax (float): Maximum step size for the ODE solver.
        step (float): Initial step size for the ODE solver.

        Returns:
        tuple: tuple containing:
            - **Mb** (*float*): The mass of the baryonic component of the neutron star in grams.
            - **Rns** (*float*): The radius of the neutron star in centimeters.
            - **Mdm_core** (*float*): The mass of the dark matter core in grams.
            - **Mdm_halo** (*float*): The mass of the dark matter halo in grams. If there is no halo, this will be zero.
            - **Rdm_core** (*float*): The radius of the dark matter core in centimeters. If there is no core, this will be zero.
            - **Rdm_halo** (*float*): The radius of the dark matter halo in centimeters. If there is no halo, this will be zero.
            - **tidal** (*float*): The tidal deformability of the neutron star. If `two_fluid_tidal` is False, this will be zero.

    
    """
    

    #Scaling the baryonic and dark matter equations of state from cgs (g/cm^3 for the energy densities and g/(cm s^2) for pressure) to geometrized units
    eps = eps* G / c**2. #eps, pres, eps_dm, pres_dm are all scaled to geometrized units
    pres = pres* G / c**4.
    eps_dm = eps_dm* G / c**2.
    pres_dm = pres_dm* G / c**4.

    # get central baryonic and dark matter pressure from central densities, which are originally in g/cm^3
    pcent = PofE(float64(epscent * G / c**2.), eps, pres)
    pcent_dm = PofE(float64(epscent_dm * G/c**2.),eps_dm,pres_dm)

    # set maxmium radius to integrate out to (in cm) inside the neutron star
    rmax = 2.5e6 # 25 km
    #Gtt = np.empty((0, 2), float)
    Array = []
    
    #Array = np.append(Array, np.array([1e-5,0.,epscent * G / c**2., pcent,epscent_dm * G/c**2., pcent_dm]),axis = 0)

    # Define the stopping criterium for the single fluid, i.e. pressure is zero
    def press_zero(r, Z, epsgrid, presgrid):
        return Z[1]
    press_zero.terminal = True
    press_zero.direction = -1

    sol_in = solve_ivp(TOV_complete, t_span=(1e-5, rmax), y0=np.array([0., 0., pcent, pcent_dm, 0.0]),
                       method='RK45', t_eval=None, args=(eps_dm, pres_dm, eps, pres), max_step=500.,atol = 1e-5, rtol = 1e-5)
    #Gtt = np.append(Gtt, np.array([sol_in.t, sol_in.y[4]]).T, axis=0)
    #print(sol_in.y[0][-1]* pow(c,2) / G)
    Pdm = sol_in.y[3][-1]
    Pb = sol_in.y[2][-1]


    for i in range(len(sol_in.y[0])):
        Array.append([sol_in.t[i], sol_in.y[0][i] + sol_in.y[1][i], EofP(sol_in.y[2][i],eps,pres), sol_in.y[2][i], EofP(sol_in.y[3][i],eps_dm,pres_dm), sol_in.y[3][i]])
    # solve for the rest of the dark matter component, i.e., there is a dark matter halo
    if(Pdm > Pb):
        if dm_halo == False:
            Mdm_halo = 0.0
            Mdm_core = sol_in.y[1][-1]
            Mdm = Mdm_core + Mdm_halo
            Rdm_halo = 999e5
            Rdm_core = 0.0
            Rns = 0.0
            Mb = sol_in.y[0][-1]
        else:
            Mb = sol_in.y[0][-1]
            Rns = sol_in.t[-1]
            Rdm_core = sol_in.t[-1]
            Mdm_core = sol_in.y[1][-1]
            rmax = 5e8 #set the maximum radius out to (in cm) through the ADM halo, up to 500 km
            sol_in2 = solve_ivp(TOV_single, t_span=(sol_in.t[-1], rmax), 
                                y0=np.array([sol_in.y[0][-1] + sol_in.y[1][-1], Pdm, sol_in.y[4][-1]]),
                                method='RK45', t_eval=None, args=(eps_dm, pres_dm), max_step=5000., events=press_zero)

            Mdm_halo = sol_in2.y[0][-1] - Mb - Mdm_core
            Rdm_halo = sol_in2.t[-1]
            #Gtt[:,1] = Gtt[:,1] - (sol_in2.y[2][-1] - 0.5*log(1 - 2 * (Mb + Mdm_core+Mdm_halo)/Rdm_halo))

            
            for i in range(len(sol_in2.y[0])):
                Array.append([sol_in2.t[i], sol_in2.y[0][i],0.0, 0.0, EofP(sol_in2.y[1][i],eps_dm,pres_dm), sol_in2.y[1][i]])

    # solve for the rest of the baryonic star, i.e., there is no dark matter halo
    if (Pb > Pdm):
        Rdm_core = sol_in.t[-1]
        Mdm_core = sol_in.y[1][-1]
        Mdm_halo = 0.0
        Rdm_halo = 0.0
        rmax = 2.5e6 #2.5e2 km
        sol_in2 = solve_ivp(TOV_single, t_span=(sol_in.t[-1], rmax), 
                            y0=np.array([sol_in.y[0][-1] + sol_in.y[1][-1], Pb+Pdm, sol_in.y[4][-1]]),
                            method='RK45', t_eval=None, args=(eps, pres), max_step=500.)
        #print(sol_in2.y[0][-1]* pow(c,2) / G)
        Mb = sol_in2.y[0][-1] - Mdm_core
        Rns = sol_in2.t[-1]
        #Gtt = np.append(Gtt, np.array([sol_in2.t, sol_in2.y[2]]).T, axis=0)
        #Gtt[:,1] = Gtt[:,1] - (sol_in2.y[2][-1] - 0.5*log(1 - 2 * (Mb + Mdm_core)/Rns))

        for i in range(len(sol_in2.y[0])):
            Array.append([sol_in2.t[i], sol_in2.y[0][i], EofP(sol_in2.y[1][i],eps,pres), sol_in2.y[1][i], 0.0, 0.0])
        
    Mb = Mb* pow(c,2) / G #scaled back into cgs units of grams
    Mdm_core = Mdm_core* pow(c,2) / G #scaled back into cgs units of grams
    Mdm_halo = Mdm_halo* pow(c,2) / G #scaled back into cgs units of grams
    Array = np.asarray(Array)

    if two_fluid_tidal == False:
        tidal = 0.
    else:
        from neost.tovsolvers.TidalDef import solveTidal
        tidal  = solveTidal(Array,dm_halo)

    return Mb, Rns, Mdm_core, Mdm_halo, Rdm_core, Rdm_halo, tidal