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

cimport numpy as np


cdef double c = global_imports._c
cdef double G = global_imports._G
cdef double Msun = global_imports._M_s

cdef double ry

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


# in terms of r
cdef double pressure_epsilon(double pressure, double eps[], double pres[], int idx) nogil:
    if idx == 0:
        eds = eps[0] * pow(pressure / pres[0], 3. / 5.)
    if idx > 0:
        ci = log(pres[idx] / pres[idx - 1]) / log(eps[idx] / eps[idx - 1])
        eds = eps[idx - 1] * pow(pressure / pres[idx - 1], 1. / ci)
    return eds

cdef double epsilon_pressure(double epsilon, double pres[], double eps[], int idx) nogil: 
    if idx==0:
        pressure = pres[0]*pow(epsilon/eps[0], 5./3.)
    if idx>0.:
        ci = log(pres[idx]/pres[idx-1])/log(eps[idx]/eps[idx-1])
        pressure = pres[idx-1] * pow(epsilon/eps[idx-1], ci)
    return pressure

cdef double pressure_adind(double pressure, double eps[], double pres[], int idx) nogil:
    if idx==0:
        eds = eps[0]*pow(pressure/pres[0], 3./5.)
        adind = 5./3. * pres[0]*pow(eds/eps[0], 5./3.) *1./eds *(eds+pressure)/pressure
    if idx>0:
        ci = log(pres[idx]/pres[idx-1])/log(eps[idx]/eps[idx-1])
        eds = eps[idx-1] * pow(pressure/pres[idx-1], 1./ci)
        adind = ci * pres[idx-1]*pow(eds/eps[idx-1], ci) *1./eds *(eds+pressure)/pressure
    return adind


cdef int TOV(double r, const double y[], double f[], void * par) noexcept nogil: # noexcept required for Cython3, it indicates that exceptions raised by this function will not be propagated to calling python functions. A warning will be printed, however.
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


def solveTOVr(double rhocent, eos_eps, eos_pres, double atol, 
              double rtol, double hmax, double step):

    cdef int i
    cdef double Pmin = 1e4 * G * pow(c,-4)

    eos_pres, indices = np.unique(np.log10(eos_pres).round(decimals=3),
                                  return_index=True)
    eos_pres = 10**eos_pres * G * pow(c,-4) #scaled to geometrized
    eos_eps = eos_eps[np.sort(indices)] * G * pow(c,-2) #scaled to geometrized

    cdef int num = len(eos_pres)
    cdef double num_double = float(num)

    # create C array from the python array
    cdef np.ndarray[double, ndim=1, mode="c"] eps_cython = np.asarray(eos_eps, dtype='float', order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] pres_cython = np.asarray(eos_pres, dtype='float', order="C")


    # Set initial conditions for solving the TOV equations 
    rhocent = rhocent * G * pow(c,-2)

    cdef int idx = binarySearch(&eps_cython[0], 0, num, rhocent)
    cdef double pcent = epsilon_pressure(rhocent, &pres_cython[0], &eps_cython[0], idx)
    cdef double adindcent = pressure_adind(pcent, &eps_cython[0], &pres_cython[0], idx)

    cdef double r = 4.441e-16

    dr, initial = initial_conditions(rhocent, pcent, adindcent)
    cdef double *stateTOV = <double*> malloc(5 * sizeof(double))
    for i in range(5):
        stateTOV[i] = initial[i]

    cdef double *params[3]
    params[0] = &eps_cython[0]
    params[1] = &pres_cython[0]
    params[2] = &num_double

    ## Initialize the ODE integrator ##
    cdef int dim=5

    cdef gsl_odeiv2_system sys = [TOV, NULL, dim, &params]
    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, r, atol, rtol)

    cdef double eps, dpdr, dmdr
    cdef double r_c = r

    Gtt = np.empty((0, 2), float)
    dalphadr = 0.0
    Gtt = np.append(Gtt, np.array([[r, dalphadr]]), axis=0)

    ## Integrate the TOV equations ##
    while (stateTOV[0]>Pmin):
        
        status = gsl_odeiv2_driver_apply(d, &r, dr, stateTOV)
        if (status != GSL_SUCCESS):
            printf ("error, return value=%d\n", status)
            break

        stateTOV[0] = sqrt(pow(stateTOV[0], 2))

        idx = binarySearch(&pres_cython[0], 0, num, stateTOV[0])
        eps = pressure_epsilon(stateTOV[0], &eps_cython[0], &pres_cython[0], idx)
        dpdr = -(eps + stateTOV[0]) * (stateTOV[1] + 4.*pi*pow(r,3) * stateTOV[0])
        dpdr = dpdr*pow(r*(r - 2.*stateTOV[1]), -1)
        dalphadr = -dpdr * pow(eps + stateTOV[0], -1)
        Gtt = np.append(Gtt, np.array([[r, stateTOV[4]]]), axis=0)
        dmdr = 4.*pi*pow(r,2) * eps
        dr = r + step * pow(pow(stateTOV[1],-1) * dmdr - pow(stateTOV[0],-1)*dpdr, -1)
    


    gsl_odeiv2_driver_free (d)


    cdef double Mb = stateTOV[1] #geometrized
    cdef double Rns = r
    cdef double y = Rns * stateTOV[3]/stateTOV[2]
    cdef double tidal = tidal_deformability(y, Mb, Rns)

    Gtt[:,1] = Gtt[:,1] - (stateTOV[4] - 0.5*log(1 - 2 * Mb/Rns))

    Mb = Mb * pow(c,2) / G #scaled back into cgs units of grams

    free(stateTOV)

    return Mb, Rns, tidal, Gtt

