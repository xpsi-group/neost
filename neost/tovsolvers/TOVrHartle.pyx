# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

from libc.math cimport pow, pi, abs, log
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free

from neost import global_imports
cdef double c = global_imports._c
cdef double G = global_imports._G

cimport numpy as np
import numpy as np

from GSL cimport *

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
    if idx>0:
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

# w refers to $\hbar\omega$ 
cdef int TOVrHartle(double r, const double y[], double f[], void *params) noexcept nogil:
    cdef double P = y[0]
    cdef double m = y[1]
    cdef double H = y[2]
    cdef double B = y[3]
    cdef double w = y[4]
    cdef double dw_dr = y[5]
    cdef double r2m
    cdef double denom
    cdef double eps
    cdef double adind

    cdef double *rhotest = (<double**> params)[0]
    cdef double *prestest = (<double**> params)[1]
    cdef double *num_double = (<double**>params)[2]
    cdef int num = <int> num_double[0]
    cdef int idx = 0

    if P <= 0:
        return GSL_EDOM

    idx = binarySearch(prestest, 0, num, P)
    eps = pressure_epsilon(P, rhotest, prestest, idx)

    denom = 1. - 2. * m / r
    if denom <= 0:
        return GSL_EDOM
    r2m = r * denom

    # TOV equations
    f[0] = - (eps + P) * (m + 4. * pi * r**2 * r * P) / (r * r2m)
    f[1] = 4. * pi * r**2 * eps

    # Tidal deformability equations
    f[2] = y[3]
    adind = pressure_adind(P, rhotest, prestest, idx)
    f[3] = 2. * H * (-2. * pi * (5. * eps + 9. * P + (eps + P)**2 / (P * adind)) + (3. / r**2) + 2. * ((m / r**2) + 4. * pi * r * P)**2 / denom) / denom \
         + 2. * B / r * (-1. + m / r + 2. * pi * r**2 * (eps - P)) / denom 

    # Hartle-Thorne equations
    f[4] = dw_dr
    f[5] = (4. * pi * r**2 * (eps + P) / r2m - (4. / r)) * dw_dr \
           + (16. * pi * r * (eps + P) / r2m) * w
    
    return GSL_SUCCESS

cdef double calc_moment_of_inertia(double w, double dw_dr, double R) nogil:
    return R**4 * dw_dr / (6 * w + 2 * R * dw_dr)

cdef double calc_tidal_deformability(double y2, double M, double R) nogil:
    cdef double C = M / R
    cdef double Eps = 4.*C**3.*(13. - 11.*y2 + C*(3.*y2 - 2.) + 2.*C**2.*(1.+y2)) + \
                        3.*(1.-2.*C)**2.*(2. - y2 + 2.*C*(y2-1.))*log(1.-2.*C) + \
                        2.*C*(6. - 3.*y2 + 3.*C*(5.*y2 - 8.))
    if Eps <= 0:
        return -1.0
        
    cdef double tidal_def = 16./(15.*Eps) *(1. - 2.*C)**2. *(2. + 2.*C*(y2-1.) - y2)
    return tidal_def

def solveTOVrHartle(double epscent,  eos_eps, eos_pres, double atol=1e-6, double rtol=1e-4, double step=0.5, double r0=1e-16):

    # setup eos grid for ODE solver
    eos_pres, indices = np.unique(np.log10(eos_pres).round(decimals=3),
                                  return_index=True)
    eos_pres = 10**eos_pres * G * pow(c,-4) 
    eos_eps = eos_eps[np.sort(indices)] * G * pow(c,-2)

    # create C array from the python array
    cdef np.ndarray[double, ndim=1, mode="c"] eps_array = np.asarray(eos_eps, dtype='float', order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] pres_array = np.asarray(eos_pres, dtype='float', order="C")

    cdef int num = len(pres_array)
    cdef double num_double = float(num)

    # pack up extra params for for the ODE solver
    cdef double *sys_params[3]
    sys_params[0] = &eps_array[0]
    sys_params[1] = &pres_array[0]
    sys_params[2] = &num_double

    # initialise ODE solver
    cdef gsl_odeiv2_system sys
    sys.function = TOVrHartle
    sys.jacobian = NULL
    sys.dimension = 6
    sys.params = <void*>sys_params

    # initial conditions
    cdef double epsilon = epscent * G * pow(c, -2)
    cdef int idx = binarySearch(&eps_array[0], 0, num, epsilon)
    cdef double Pc = epsilon_pressure(epsilon, &pres_array[0], &eps_array[0], idx)
    cdef double r = r0
    cdef double m0 = 4. / 3. * pi * r * r * r * epsilon
    cdef double P0 = Pc - (2. * pi / 3.) * (Pc + epsilon) * (3. * Pc + epsilon) * r * r
    cdef double w0 = 1.0
    cdef double dw0dr = 0.0
    cdef double H0 = r**2
    cdef double B0 = 2. * r

    # split because cython complains
    cdef double *stateTOV = <double*> malloc(6 * sizeof(double))
    stateTOV[0] = P0
    stateTOV[1] = m0
    stateTOV[2] = H0
    stateTOV[3] = B0
    stateTOV[4] = w0
    stateTOV[5] = dw0dr

    # create ODE driver
    # rk8pd seems to perform well
    cdef gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, r, atol, rtol)
    
    # integrate until surface (Pmin) reached
    cdef int status
    cdef double Pmin = 1e4 * G / (c * c * c * c)

    cdef double r_next = r + 10
    cdef double dpdr, dmdr, r2m, eps, adind
    while (stateTOV[0] > Pmin):
        status = gsl_odeiv2_driver_apply(driver, &r, r_next, stateTOV)
        
        if status != GSL_SUCCESS:
            printf("Error: %d\n", status)
            return [0.0, 0.0, 0.0, 0.0]

        # some core pressures might not converge properly
        if r > 2e6: # 20 km
            break

        idx = binarySearch(&pres_array[0], 0, num, stateTOV[0])
        eps = pressure_epsilon(stateTOV[0], &eps_array[0], &pres_array[0], idx)

        r2m = r - 2. * stateTOV[1]
        if r2m <=0:
            break
        
        dpdr = - (eps + stateTOV[0]) * (stateTOV[1] + 4 * pi * r * r * r * stateTOV[0]) / (r * r2m)
        dmdr = 4 * pi * r * r * eps

        r_next = r + step / (dmdr / stateTOV[1] - dpdr / stateTOV[0])

    cdef double I = calc_moment_of_inertia(stateTOV[4], stateTOV[5], r)
    cdef double y = r * stateTOV[3] / stateTOV[2]
    cdef double tidal = calc_tidal_deformability(y, stateTOV[1], r)

    # M, R, tidal, I
    # return in cgs
    result = 4 * [0.0]
    result[0] = stateTOV[1] * c**2 / G
    result[1] = r
    result[2] = tidal 
    result[3] = I * c**2 / G

    # free all the ram
    gsl_odeiv2_driver_free(driver)
    free(stateTOV)

    return result