# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

from libc.math cimport sqrt, sin, cos, acos, log10, pow, exp, pi, log, floor
from libc.stdio cimport printf, setbuf, stdout
from libc.stdlib cimport malloc, free, abs

import numpy as np

from GSL cimport *
from .. import global_imports
from cython.operator import dereference
from matplotlib import pyplot
import time
cimport numpy as np


cdef double c = global_imports._c
cdef double G = global_imports._G
cdef double Msun = global_imports._M_s



cdef int binarySearch(double arr[], int low, int high, double key) nogil:
    cdef int mid 
    while high > low:
        mid = (low + high) / 2
        if low > high:
            break
        if arr[mid] > key:
            high = mid
        else:
            low = mid + 1
    return low


# in terms of r epsilon(pressure)
cdef double pressure_epsilon(double pressure, double eps[], double pres[], int idx) nogil:
    if idx == 0:
        eds = eps[0] * pow(pressure / pres[0], 3. / 5.)
    if idx > 0:
        ci = log(pres[idx] / pres[idx - 1]) / log(eps[idx] / eps[idx - 1])
        eds = eps[idx - 1] * pow(pressure / pres[idx - 1], 1. / ci)
    return eds

#pressure(epsilon)
cdef double epsilon_pressure(double epsilon, double pres[], double eps[], int idx) nogil: 
    if idx==0:
        pressure = pres[0]*pow(epsilon/eps[0], 5./3.)
    if idx>0.:
        ci = log(pres[idx]/pres[idx-1])/log(eps[idx]/eps[idx-1])
        pressure = pres[idx-1] * pow(epsilon/eps[idx-1], ci)
    return pressure

#adind(pressure)
cdef double pressure_adind(double pressure, double eps[], double pres[], int idx) nogil:
    if idx==0:
        eds = eps[0]*pow(pressure/pres[0], 3./5.)
        adind = 5./3. * pres[0]*pow(eds/eps[0], 5./3.) *1./eds *(eds+pressure)/pressure
    if idx>0:
        ci = log(pres[idx]/pres[idx-1])/log(eps[idx]/eps[idx-1])
        eds = eps[idx-1] * pow(pressure/pres[idx-1], 1./ci)
        adind = ci * pres[idx-1]*pow(eds/eps[idx-1], ci) *1./eds *(eds+pressure)/pressure
    return adind




cdef int TOV_single(double r, const double y[], double f[], void * par) noexcept nogil: # noexcept required for Cython3, it indicates that exceptions raised by this function will not be propagated to calling python functions. A warning will be printed, however.
#All inputs are assumed to be in geometrized units
    cdef double p
    cdef double eps
    cdef double ad_index

    cdef double *rhotest = (<double**> par)[0]
    cdef double *prestest = (<double**> par)[1]
    cdef double *num_double = (<double**>par)[2]
    cdef int num = <int> num_double[0]
    p = sqrt(y[0]*y[0])
    cdef int idx = binarySearch(prestest, 0, num, p)
    eps = pressure_epsilon(p, rhotest, prestest, idx)
    ad_index = pressure_adind(p, rhotest, prestest, idx)

    f[0] = -(eps + p) * (y[1] + 4. * pi * pow(r,3) * p)
    f[0] *= pow(r * (r - 2.*y[1]), -1)
    f[1] = 4.*pi*pow(r,2) * eps
    
    f[2] = y[3]
    f[3] = 2.*pow(1.-2.*y[1]/r, -1) *y[2]*(-2.*pi*(5.*eps + 9.*p+(eps+p)**2./(p*ad_index)) +3./pow(r,2) + 2.*pow(1.-2.*y[1]/r,-1)*pow(y[1]/pow(r,2) + 4.*pi*r*p,2)) \
           + 2.*y[3]/r *pow(1. - 2.*y[1]/r, -1)*(-1. + y[1]/r + 2.*pi*pow(r,2)*(eps-p))

    f[4] = (y[1] + 4.*pi*pow(r,3) * p) * pow(r*(r - 2.*y[1]), -1)

    return GSL_SUCCESS




cdef double Q22(double x) nogil:
    return 3./2. *(pow(x,2) - 1.) * log((x+1.)/(x-1.)) - (3.*pow(x,3)- 5.*x)/(pow(x,2) - 1.)

cdef double Q21(double x) nogil:
    return sqrt(pow(x,2) - 1.) *((3.*pow(x,2) - 2.)/(pow(x,2) - 1.) - 3.*x/2. *log((x+1.)/(x-1.)))


def initial_conditions(double rhocent, double pcent, adindcent=2.):
        """
        Set the initial conditions for solving the structure equations. 

        Args: 
            eos (object): An object that takes energy density as input and outputs pressure, both in geometrized units.
            w0 (float): The initial value of the rotational drag. Not known a priori, but can be calculated after the TOV equations are solved.
            j0 (float): The initial value of j. Not known a priori, but can be calculated after the TOV equations are solved.
            static (bool): Calculate initial conditions for a static star (True) or a rotating star (False). 

        Returns:
            tuple: tuple containing:

                - **dr** (*float*): The initial stepsize in cm. 
                - **intial**   (*array*): A np array storing the initial conditions.

        """
        cdef double r = 4.441e-16
        cdef double dr = 10.

        cdef double P0, m0, y20

        P0 = pcent - (2.*pi/3.)*(pcent + rhocent) *(3.*pcent + rhocent)*r**2.
        m0 = 4./3. *pi *rhocent*r**3.
        # y20 = 2.*(1. - 2.*pi/7.*(rhocent/3. + 11.*pcent + pow(rhocent+pcent,2.)/(pcent*adindcent))*r**2.)
        h0 = r**2.
        b0 = 2.*r

        initial = np.array([P0, m0, h0, b0, 0.0])

        return dr, initial

cdef double tidal_deformability(double y2, double Mns, double Rns) nogil:
#Assumed to be in geometrized
    cdef double C = Mns/Rns
    cdef double Eps = 4.*C**3.*(13. - 11.*y2 + C*(3.*y2 - 2.) + 2.*C**2.*(1.+y2)) + \
                        3.*(1.-2.*C)**2.*(2. - y2 + 2.*C*(y2-1.))*log(1.-2.*C) + \
                        2.*C*(6. - 3.*y2 + 3.*C*(5.*y2 - 8.))
    cdef double tidal_def = 16./(15.*Eps) *(1. - 2.*C)**2. *(2. + 2.*C*(y2-1.) - y2)
    return tidal_def

def solveTOVde(double rhocent, double rho_plus, double alpha, eos_epsb, eos_presb, eos_epsde, eos_presde, double atol,
              double rtol, double hmax, double step):

    #cdef double atol=1e-6
    #cdef double rtol=1e-4
    #cdef double hmax=1000.
    #cdef double step= 0.46

    cdef int i
    cdef double Pmin = 1e2 * G * pow(c,-4) 

      # Making sure there are no double values in dark matter pressure and energy density arrays
    eos_presde, indices = np.unique(np.log10(eos_presde).round(decimals=5),
                                  return_index=True)

    eos_presde = 10**(eos_presde) * G * pow(c,-4) #scaled to geometrized
    eos_epsde = eos_epsde[np.sort(indices)] * G * pow(c,-2) #scaled to geometrized

    cdef int numde = len(eos_presde)
    cdef double numde_double = float(numde)  

    # Making sure there are no double values in baryonic pressure and energy density arrays
    eos_presb, indices = np.unique(np.log10(eos_presb).round(decimals=5),
                                  return_index=True)
    eos_presb = 10**eos_presb * G * pow(c,-4) #scaled to geometrized
    eos_epsb = eos_epsb[np.sort(indices)] * G * pow(c,-2) #scaled to geometrized

    cdef int numb = len(eos_presb)
    cdef double numb_double = float(numb)



    # create C array from the python array
    cdef np.ndarray[double, ndim=1, mode="c"] epsb_cython = np.asarray(eos_epsb, dtype='float', order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] presb_cython = np.asarray(eos_presb, dtype='float', order="C")

    cdef np.ndarray[double, ndim=1, mode="c"] epsde_cython = np.asarray(eos_epsde, dtype='float', order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] presde_cython = np.asarray(eos_presde, dtype='float', order="C")



    # Set initial conditions for solving the TOV equations
    rhocent = rhocent * G * pow(c,-2)

    #alpha = rho_minus/rho_plus -- > alpha * rho_plus = rho_minus
    rho_minus = alpha * rho_plus * G * pow(c,-2)
    rho_plus = rho_plus * G * pow(c,-2)

    cdef int idx = binarySearch(&epsde_cython[0], 0, numde, rhocent)
    cdef double pcent = epsilon_pressure(rhocent, &presde_cython[0], &epsde_cython[0], idx)
    cdef double adindcent = pressure_adind(pcent, &epsde_cython[0], &presde_cython[0], idx)

    cdef int idx_plus = binarySearch(&epsde_cython[0], 0, numde, rho_plus)
    cdef double P_plus = epsilon_pressure(rho_plus, &presde_cython[0], &epsde_cython[0], idx_plus)


    cdef double r = 4.441e-16

    dr, initial = initial_conditions(rhocent, pcent, adindcent)


    cdef double *stateTOV = <double*> malloc(5 * sizeof(double))
    for i in range(5):
        stateTOV[i] = initial[i]

    cdef double *params[3]
    params[0] = &epsde_cython[0]
    params[1] = &presde_cython[0]
    params[2] = &numde_double

    ## Initialize the ODE integrator ##
    cdef int dim=5

    cdef gsl_odeiv2_system sys = [TOV_single, NULL, dim, &params]
    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, r, atol, rtol) #rk8pd

    cdef double eps, dpdr, dmdr
    cdef double r_c = r

    Gtt = np.empty((0, 2), float)
    dalphadr = 0.0
    Gtt = np.append(Gtt, np.array([[r, dalphadr]]), axis=0)

    #stateTOV[0] is the pressure, stateTOV[1] is the mass, the others relate to the tidal def (stateTOV[2,3]), and stateTOV[4] is the alpha metric function
   ## Integrate the TOV equations ##
    while (stateTOV[0]>P_plus):
        
        status = gsl_odeiv2_driver_apply(d, &r, dr, stateTOV)
        if (status != GSL_SUCCESS):
            printf ("error, return value=%d\n", status)
            break

        stateTOV[0] = sqrt(pow(stateTOV[0], 2))

        idx = binarySearch(&presde_cython[0], 0, numde, stateTOV[0])
        eps = pressure_epsilon(stateTOV[0], &epsde_cython[0], &presde_cython[0], idx)
        dpdr = -(eps + stateTOV[0]) * (stateTOV[1] + 4.*pi*pow(r,3) * stateTOV[0])
        dpdr = dpdr*pow(r*(r - 2.*stateTOV[1]), -1)
        dalphadr = -dpdr * pow(eps + stateTOV[0], -1)
        Gtt = np.append(Gtt, np.array([[r, stateTOV[4]]]), axis=0)
        dmdr = 4.*pi*pow(r,2) * eps
        dr = r + step * pow(pow(stateTOV[1],-1) * dmdr - pow(stateTOV[0],-1)*dpdr, -1)


    cdef double Mde_core = stateTOV[1]
    cdef double P_dis = stateTOV[0] #geometrized
    cdef double Rde_core = r


    


    gsl_odeiv2_driver_free (d)
    cdef double *params_single[3]
    cdef int dim_single = 5

    params_single[0] = &epsb_cython[0] 
    params_single[1] = &presb_cython[0] 
    params_single[2] = &numb_double 


    cdef gsl_odeiv2_system sys_single = [TOV_single, NULL, dim_single, &params_single]
    cdef gsl_odeiv2_driver * d_single = gsl_odeiv2_driver_alloc_y_new(&sys_single, gsl_odeiv2_step_rkf45, r, atol, rtol)
    cdef double *stateTOV_single = <double*> malloc(5 * sizeof(double))



    stateTOV_single[0] = stateTOV[0]# Baryonic pressure
    stateTOV_single[1] = stateTOV[1] #Mass
    stateTOV_single[2] = stateTOV[2]
    stateTOV_single[3] = stateTOV[3]
    stateTOV_single[4] = stateTOV[4] # alpha

        ## Integrate the TOV equations ##
    while (stateTOV_single[0]>Pmin):
        
        status = gsl_odeiv2_driver_apply(d_single, &r, dr, stateTOV_single)
        if (status != GSL_SUCCESS):
            printf ("error, return value=%d\n", status)
            break

        stateTOV_single[0] = sqrt(pow(stateTOV_single[0], 2))


        idx = binarySearch(&presb_cython[0], 0, numb, stateTOV_single[0])
        eps = pressure_epsilon(stateTOV_single[0], &epsb_cython[0], &presb_cython[0], idx)

        dpdr = -(eps + stateTOV_single[0]) * (stateTOV_single[1] + 4.*pi*pow(r,3) * stateTOV_single[0])
        dpdr = dpdr*pow(r*(r - 2.*stateTOV_single[1]), -1)
        dalphadr = -dpdr * pow(eps + stateTOV_single[0], -1)
        Gtt = np.append(Gtt, np.array([[r, stateTOV_single[4]]]), axis=0)
        dmdr = 4.*pi*pow(r,2) * eps
        dr = r + step * pow(pow(stateTOV_single[1],-1) * dmdr - pow(stateTOV_single[0],-1)*dpdr, -1)



    gsl_odeiv2_driver_free (d_single)
    cdef double Mb = stateTOV_single[1] #geometrized
    cdef double Rns = r
    cdef double y = Rns * stateTOV_single[3]/stateTOV_single[2]
    cdef double tidal = tidal_deformability(y, Mb, Rns)


    Gtt[:,1] = Gtt[:,1] - (stateTOV_single[4] - 0.5*log(1 - 2 * Mb/Rns))

    Mb = Mb * pow(c,2) / G #scaled back into cgs units of grams
    Mde_core = Mde_core * pow(c,2) / G #scaled back into cgs units of grams

    Mb = Mb - Mde_core


    free(stateTOV)
    free(stateTOV_single)


    return Mb, Mde_core, Rde_core, Rns, tidal, Gtt