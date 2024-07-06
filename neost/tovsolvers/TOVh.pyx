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
from scipy.interpolate import UnivariateSpline

cimport numpy as np


cdef double c = global_imports._c
cdef double G = global_imports._G
cdef double Msun = global_imports._M_s

cdef double ry

cdef int binarySearch(double arr[], int low, int high, double key) nogil:
    cdef int mid 
    for i in range(high):
        mid = (low + high) / 2
        if low > high:
            break
        if arr[mid] > key:
            high = mid
        else: 
            low = mid + 1
    return low

# in terms of h
cdef double h_epsilon(double hvar, double eps[], double h[], double pres[], int idx) nogil:
    if idx == 0:
        eds = eps[0] * pow(eps[0] / pres[0] * (exp(2. * hvar / 5.) - 1.), 3. / 2.)
    if idx > 0:
        ci = log(pres[idx] / pres[idx - 1]) / log(eps[idx] / eps[idx - 1])
        eds = eps[idx - 1] * pow((eps[idx - 1] + pres[idx - 1]) / pres[idx - 1] * exp((ci - 1.) / ci * (hvar - h[idx - 1])) - eps[idx - 1] / pres[idx - 1], 1. / (ci - 1.))
    return eds

cdef double h_pressure(double hvar, double eps[], double h[], double pres[], int idx) nogil:
    if idx == 0:
        eds = eps[0] * pow(eps[0] / pres[0] * (exp(2. * hvar / 5.) - 1.), 3. / 2.)
        pressure = pres[0] * pow(eds / eps[0], 5. / 3.)
    if idx > 0:
        ci = log(pres[idx] / pres[idx - 1]) / log(eps[idx] / eps[idx - 1])
        eds = eps[idx - 1] * pow((eps[idx - 1] + pres[idx - 1]) / pres[idx - 1] * exp((ci - 1.) / ci * (hvar - h[idx - 1])) - eps[idx - 1] / pres[idx - 1], 1. / (ci - 1.))
        pressure = pres[idx - 1] * pow(eds / eps[idx - 1],ci)
    return pressure

cdef double h_adind(double hvar, double eps[], double h[], double pres[], int idx) nogil:
    if idx==0:
        eds = eps[0]*pow(eps[0]/pres[0]*(exp(2.*hvar/5.)-1.), 3./2.)
        pressure = pres[0]*pow(eds/eps[0], 5./3.)
        adind = 5./3. * pres[0]*pow(eds/eps[0],5./3.) *1./eds *(eds+pressure)/pressure
    if idx>0:
        ci = log(pres[idx]/pres[idx-1])/log(eps[idx]/eps[idx-1])
        eds = eps[idx-1]*pow((eps[idx-1]+pres[idx-1])/pres[idx-1] * exp((ci -1.)/ci *(hvar - h[idx-1])) - eps[idx-1]/pres[idx-1], 1./(ci -1.))
        pressure = pres[idx-1]*pow(eds/eps[idx-1],ci)
        adind = ci * pres[idx-1]*pow(eds/eps[idx-1],ci) *1./eds *(eds+pressure)/pressure
    return adind


cdef int TOV_h(double hvar, const double y[], double f[], void *par) noexcept nogil: # noexcept required for Cython3, it indicates that exceptions raised by this function will not be propagated to calling python functions. A warning will be printed, however.

    cdef double eps, ad_index, p

    cdef double *rhotest = (<double**> par)[0]
    cdef double *prestest = (<double**> par)[1]
    cdef double *h = (<double**> par)[2]
    cdef double *num_double = (<double**>par)[3]
    cdef int num = <int> num_double[0]
    
    cdef int idx = binarySearch(h, 0, num, hvar)
    eps = h_epsilon(hvar, rhotest, h, prestest, idx)
    p = h_pressure(hvar, rhotest, h, prestest, idx)

    ad_index = h_adind(hvar, rhotest, h, prestest, idx)
    

    f[0] = - 4.*pi*pow(y[1],3)*eps * (y[1]-2*y[0])/(y[0]+4.*pi*pow(y[1],3)*p) ## dmdh
    f[1] = f[0] *pow(4.*pi*pow(y[1],2)*eps,-1) ## drdh
    
    
    f[2] = (-y[2]**2./y[1] - (y[1] + 4.*pi*y[1]**3.*(p-eps)) * y[2]/(y[1]*(y[1]-2.*y[0])) + 4.*(y[0]+4.*pi*y[1]**3.*p)**2./(y[1]*(y[1]-2.*y[0])**2.) \
        + 6./(y[1]-2.*y[0]) - 4.*pi*y[1]**2./(y[1]-2.*y[0]) *(5.*eps + 9.*p + (eps+p)**2./(p*ad_index)))*f[1]

    # f[2] = f[1]*(y[2]/y[1] - pow(y[2],2)/y[1] - y[2]*(2/y[1] - (2*y[0] + 4.*pi*pow(y[1],3)*(p-eps))/(y[1]*(y[1]-2.*y[0]))) \
    #        - y[1]/(y[1]-2.*y[0])*(4.*pi*y[1]*(5.*eps + 9*p + pow(eps+p,2.)/(p*ad_index)) - 6./y[1] - \
    #                     4*pow(y[0]+4.*pi*pow(y[1],3)*p, 2.)/(pow(y[1],2)*(y[1]-2.*y[0]))))
    
    return GSL_SUCCESS



    

cdef double Q22(double x) nogil:
    return 3./2. *(pow(x,2) - 1.) * log((x+1.)/(x-1.)) - (3.*pow(x,3)- 5.*x)/(pow(x,2) - 1.)

cdef double Q21(double x) nogil:
    return sqrt(pow(x,2) - 1.) *((3.*pow(x,2) - 2.)/(pow(x,2) - 1.) - 3.*x/2. *log((x+1.)/(x-1.)))

def h_initial_condition (double central_h, double central_P, double central_e, double central_Gamma):

    # h = np.linspace(central_h, 0.0, 10000)

    h_par = 0.99999*central_h#h[1]
    r1_par = pow( 3./(2. * pi * ( central_e + 3. * central_P ) ), 1./2)
    r3_par = - r1_par / ( 4. * (central_e + 3. * central_P) ) * ( central_e - 3. * central_P - ( 3. * pow( central_e + central_P, 2. ) ) / ( 5. * central_P * central_Gamma ) )
    r_c = r1_par * pow( central_h - h_par, 1./2 ) + r3_par * pow( central_h - h_par, 3./2 );


    m3_par = (4. * pi / 3) * central_e * pow( r1_par, 3. )
    m5_par = 4. * pi* pow( r1_par, 3. ) * ( central_e * r3_par / r1_par - pow( central_e + central_P, 2 ) / ( 5. * central_P * central_Gamma ) )
    m_c = m3_par * pow( central_h - h_par, 3./2 ) + m5_par * pow( central_h - h_par, 5./2 )

    y2_par = - 6. / ( 7. * (central_e + 3. * central_P) ) * ( central_e / 3. + 11. * central_P + pow( central_e + central_P, 2. ) / ( central_P * central_Gamma ) )
    y_c = 2. + y2_par * ( central_h - h_par );
    
    initial = np.array([m_c, r_c, y_c])

    return h_par, initial


cdef double tidal_deformability(double y2, double Mns, double Rns) nogil:

    cdef double C = Mns/Rns
    cdef double Eps = 4.*C**3.*(13. - 11.*y2 + C*(3.*y2 - 2.) + 2.*C**2.*(1.+y2)) + \
                        3.*(1.-2.*C)**2.*(2. - y2 + 2.*C*(y2-1.))*log(1.-2.*C) + \
                        2.*C*(6. - 3.*y2 + 3.*C*(5.*y2 - 8.))
    cdef double tidal_def = 16./(15.*Eps) *(1. - 2.*C)**2. *(2. + 2.*C*(y2-1.) - y2)
    return tidal_def


def solveTOVh(double rhocent, eos_eps, eos_pres, double atol=1e-6, 
              double rtol=1e-5, double hmax=1000., double step=0.46):

    cdef int i
    rhocent = rhocent * G * pow(c,-2)

    cdef int num = len(eos_pres)
    cdef double num_double = float(num)

    # create C array from the python array ##
    cdef np.ndarray[double, ndim=1, mode="c"] eps_cython = np.asarray(eos_eps * G * pow(c,-2), dtype='float', order="C") #scaled into geometrized
    cdef np.ndarray[double, ndim=1, mode="c"] pres_cython = np.asarray(eos_pres * G * pow(c,-4), dtype='float', order="C") #scaled into geometrized

    cdef double *h= <double*> malloc(num * sizeof(double))
    cdef double ci
    h[0] = 5./2.*log((eps_cython[0]+pres_cython[0])/eps_cython[0])
    for i in range(1, num):
        ci = log(pres_cython[i]/pres_cython[i-1])/log(eps_cython[i]/eps_cython[i-1])
        h[i] = h[i-1] + ci/(ci-1.) *log(eps_cython[i-1]*(eps_cython[i] + pres_cython[i])/(eps_cython[i]*(eps_cython[i-1]+pres_cython[i-1])))
    
    cdef gsl_interp_accel *hrho_acc = gsl_interp_accel_alloc()
    cdef gsl_spline *hrho = gsl_spline_alloc(gsl_interp_linear, num)
    gsl_spline_init(hrho, &eps_cython[0], &h[0], num)
    cdef double hcent = gsl_spline_eval(hrho, rhocent, hrho_acc)

    cdef int idx = binarySearch(&h[0], 0, num, hcent)
    cdef double epscent = h_epsilon(hcent, &eps_cython[0], &h[0], &pres_cython[0], idx)
    cdef double pcent = h_pressure(hcent, &eps_cython[0], &h[0], &pres_cython[0], idx)
    cdef double adindcent = h_adind(hcent, &eps_cython[0], &h[0], &pres_cython[0], idx)


    cdef double hvar = hcent
    
    dh, initial = h_initial_condition(hcent, pcent, epscent, adindcent)
    cdef double *stateTOV = <double*> malloc(3 * sizeof(double))
    for i in range(3):
        stateTOV[i] = initial[i]

    cdef double *params[4]
    params[0] = &eps_cython[0]
    params[1] = &pres_cython[0]
    params[2] = &h[0]
    params[3] = &num_double

    ## Initialize the ODE integrator ##
    cdef int dim=3
    cdef np.ndarray[double, ndim=1, mode="c"] harray = np.asarray(np.logspace(np.log10(hcent), np.log10(h[0]), 500), dtype='float', order="C")
    hvar = harray[0]

    cdef gsl_odeiv2_system sys = [TOV_h, NULL, dim, &params]
    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk4, harray[1]-harray[0], atol, rtol)

    yevolve = np.zeros((100000, 2))
    yevolve[0] = hvar, stateTOV[2]
    i = 0


    ## Integrate the TOV equations ##
    # while (hvar>h[0]):
    for i in range(499):

        status = gsl_odeiv2_driver_apply(d, &hvar, harray[i+1], stateTOV)
        if (status != GSL_SUCCESS):
            printf ("error, return value=%d\n", status)
            break
        
        # yevolve[i] = r, stateTOV[2]

    # print(initial[2], stateTOV[1]*c**2./G/Msun, stateTOV[2])
    # fig, ax = pyplot.subplots(1,1)
    # ax.plot(yevolve[:,0]/1e5, yevolve[:,1])
    # pyplot.show()

    gsl_odeiv2_driver_free (d)


    cdef double Mb = stateTOV[0] #geometrized
    cdef double Rns = stateTOV[1]
    cdef double tidal = tidal_deformability(stateTOV[2], Mb, Rns)

    Mb = Mb * pow(c,2) / G #scaled back into grams


    free(stateTOV)
    gsl_spline_free (hrho)
    gsl_interp_accel_free (hrho_acc)
    free(h)

    return Mb, Rns, tidal

