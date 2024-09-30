#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline,interp1d
from numba import jit,njit,float64,boolean
import warnings



@jit(float64(float64, float64[:], float64[:]), nopython=True)
def EofP(P, epsgrid, presgrid):
    epsgrid = epsgrid[::-1]
    presgrid = presgrid[::-1]
    idx = np.searchsorted(presgrid, P)
    if idx == 0:
        eds = epsgrid[0] * pow(P / presgrid[0], 3. / 5.)
    if idx == len(presgrid):
        eds = epsgrid[-1] * pow(P / presgrid[-1], 3./5.)
    else:
        ci = np.log(presgrid[idx] / presgrid[idx-1]) / np.log(epsgrid[idx] / epsgrid[idx-1])
        eds = epsgrid[idx-1] * pow(P / presgrid[idx-1], 1. / ci)
       
    return eds

@jit(float64(float64, float64[:], float64[:]), nopython=True)
def PofE(E, epsgrid, presgrid):
    epsgrid = epsgrid[::-1]
    presgrid = presgrid[::-1]
    idx = np.searchsorted(epsgrid, E)
    if idx == 0:
        pres = presgrid[0] * pow(E / epsgrid[0], 5. / 3.)
    if idx == len(epsgrid): 
        pres = presgrid[-1] * pow(E / epsgrid[-1], 5. / 3.)
    else:
        ci = np.log(presgrid[idx] / presgrid[idx - 1]) / np.log(epsgrid[idx] / epsgrid[idx - 1])
        pres = presgrid[idx - 1] * (E / epsgrid[idx - 1])**ci
    
    return pres


# This will be the block that will break up the array made by TOVdm
#Array will r in cm, total mass (M) in cm and then the rest are in their geometrized units form (length^-2)

def solveTidal(Array,dm_halo):
    """
    Function that solves the two-fluid tidal deformability equations. 


    Parameters
    ----------
    Array: array
        Output array that has the full TOVdm.pyx solver's radial, mass and pressure distributions of baryonic matter and ADM, respectively. 
        The units are geometrized.
    dm_halo: bool
        If True, ADM halos will be considered in the two-fluid tidal deformability calculation.

    Methods
    -------
    dp_deps(eps,deps,epsgrid,presgrid)
        Numerical derivative calculator of the derivative of pressure w.r.t energy density
    dalpha_dr(r)
        The dalpha_dr term in the TOV equations, which is related to the g_tt component of the spherically symmetric static metric.
    g_rr(r)
        g_rr component of the spherically symmetric static metric.
    dy_dr(r,y)
         yR = y(R) is related to the quadrupolar perturbed metric function and 
         is obtained as a solution to the differential equation dy_dr
    tidal_deformability(y2, Mns, Rns)
        Two-fluid deformability calculation function.

    """
    depsb =  Array[:,2][0]*10**-10
    depsdm = Array[:,4][0]*10**-10
    
    radius = Array[:,0]
    R = radius[len(radius)-1]
    Mass_Total = Array[:,1]
    Mns = Mass_Total[len(Mass_Total)-1]
    epsb_array = Array[:,2]
    Pb_array = Array[:,3]
    epsdm_array = Array[:,4]
    Pdm_array = Array[:,5]
    
    Pressures_total = Pb_array + Pdm_array
    
    
    def dp_deps(eps,deps,epsgrid,presgrid):
        if eps <=0:
            deriv = 0
        else:
            deriv = (PofE(eps+deps/2,epsgrid,presgrid)-PofE(eps-deps/2,epsgrid,presgrid))/deps
        return deriv
    
    
    def dalpha_dr(r):
        return ((4*np.pi*r**3*P(r)+M(r))/(r*(r-2*M(r))))
    
    
    def g_rr(r):
        return (pow((1-2*M(r)/r),-1))

    
    def dy_dr(r,y):
        grr = g_rr(r)
        dalphadr = dalpha_dr(r)
        epsb = epsb_r(r) 
        pb = pb_r(r) 
        epsdm = epsdm_r(r) 
        pdm =pdm_r(r) 
    

        dpdm_depsdm = dp_deps(epsdm,depsdm,epsdm_array,Pdm_array)
        dpb_depsb = dp_deps(epsb,depsb,epsb_array,Pb_array)
        
    
        P = pb+pdm
        Eps = epsb+epsdm
    
        T1 = -y**2./r
        T2 = -grr*y/r
        T3 = 4*np.pi*r*grr*y*(Eps-P)
    
        if epsb !=0 and epsdm !=0:
            deriv = (epsb+pb)/dpb_depsb + (epsdm+pdm)/dpdm_depsdm
        
        if epsdm == 0:
            deriv = (epsb+pb)/dpb_depsb
        
        
        if epsb == 0:
            deriv = (epsdm+pdm)/dpdm_depsdm
        
        T4 = -4*np.pi*r*grr*(5.*Eps+9.*P+deriv)
        T5 = 6./r*grr + 4.*r*dalphadr**2
        dydr = T1 + T2 + T3 + T4 + T5
        return dydr
    
    def tidal_deformability(y2, Mns, Rns):
        C = Mns/Rns
        Eps = 4.*C**3.*(13. - 11.*y2 + C*(3.*y2 - 2.) + 2.*C**2.*(1.+y2)) + 3.*(1.-2.*C)**2.*(2. - y2 + 2.*C*(y2-1.))*np.log(1.-2.*C) +2.*C*(6. - 3.*y2 + 3.*C*(5.*y2 - 8.))
        tidal_def = 16./(15.*Eps) *(1. - 2.*C)**2. *(2. + 2.*C*(y2-1.) - y2)
        return tidal_def

    try:
        M = UnivariateSpline(radius,Mass_Total,k=1,s=0,ext=1) 
        P = UnivariateSpline(radius,Pressures_total,k=1,s=0,ext=1)
                
        epsb_r = UnivariateSpline(radius,epsb_array,k=1,s=0,ext=1)
        pb_r = UnivariateSpline(radius,Pb_array,k=1,s=0,ext=1)
        epsdm_r = UnivariateSpline(radius,epsdm_array,k=1,s=0,ext=1)
        pdm_r =  UnivariateSpline(radius,Pdm_array,k=1,s=0,ext=1)
                
    except ValueError:
        M = interp1d(radius,Mass_Total,kind = 'linear',bounds_error = False,fill_value = [0]) 
        P = interp1d(radius,Pressures_total,kind = 'linear',bounds_error = False,fill_value = [0])
        epsb_r = interp1d(radius,epsb_array,kind = 'linear',bounds_error = False,fill_value = [0])
        pb_r = interp1d(radius,Pb_array,kind = 'linear',bounds_error = False,fill_value = [0])
        epsdm_r = interp1d(radius,epsdm_array,kind = 'linear',bounds_error = False,fill_value = [0])
        pdm_r =  interp1d(radius,Pdm_array,kind = 'linear',bounds_error = False,fill_value = [0])

    if dm_halo == False:
        if R == 0:
            tidal_def = 0
           
#Note: We have the try-except statements to perform a switch between the interpolation methods. This is because sometimes the radius array
# is monotonically increasing and not strictly increasing as UnivariateSpline requires. However, univariatespline is much faster than 
# interp1d, so I wanted to the code the switch between the two methods depending on if radius is monotonically increasing or strictly increasing, 
        else:             
            ysolution = solve_ivp(dy_dr,(radius[0],R),[2.0], method = 'RK45',atol = 0.00001, rtol = 0.00001)
            y2 = ysolution.y[0][-1]
            tidal_def = tidal_deformability(y2,Mns,R)
            
    else:
                    
        ysolution = solve_ivp(dy_dr,(radius[0],R),[2.0], method = 'RK45',atol = 0.00001, rtol = 0.00001)
        y2 = ysolution.y[0][-1]
        tidal_def = tidal_deformability(y2,Mns,R)
        
    
    return tidal_def






