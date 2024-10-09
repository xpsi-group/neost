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