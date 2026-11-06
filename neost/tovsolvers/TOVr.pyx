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
    """Perform a binary search to find the index of the largest value in `arr` that is less than or equal to `key`.

    Args:
        arr (double[]): A sorted array of double values.
        low (int): The starting index of the search range.
        high (int): The ending index of the search range.
        key (double): The value to search for.

    Returns:
        int: The index of the largest value in `arr` that is less than or equal to `key`. If `key` is smaller than the smallest value in `arr`, returns 0. If `key` is larger than the largest value in `arr`, returns `high`.
    """
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
    """
    Calculate the energy density corresponding to a given pressure. This function uses a piecewise power-law 
    interpolation based on the provided `eps` and `pres` arrays, which represent the energy density and pressure 
    values of the equation of state (EOS) grid, respectively. The `idx` parameter indicates the index in the EOS 
    grid that corresponds to the given pressure.
    Args:        
        pressure (double): The pressure for which to calculate the energy density, in geometrized units (g/(cm s^2) converted to g/(cm s^2)).
        eps (double): An array of energy density values corresponding to the EOS grid, in geometrized units (g/cm^3 converted to g/cm).
        pres (double): An array of pressure values corresponding to the EOS grid, in geometrized units (g/(cm s^2) converted to g/(cm s^2)).
        idx (int): The index in the EOS grid that corresponds to the given pressure. This index is obtained using a binary search on the `pres` array.

    Returns:
        double: The energy density corresponding to the given pressure, calculated using piecewise power-law interpolation
    """

    if idx == 0:
        eds = eps[0] * pow(pressure / pres[0], 3. / 5.)
    if idx > 0:
        ci = log(pres[idx] / pres[idx - 1]) / log(eps[idx] / eps[idx - 1])
        eds = eps[idx - 1] * pow(pressure / pres[idx - 1], 1. / ci)
    return eds

cdef double epsilon_pressure(double epsilon, double pres[], double eps[], int idx) nogil: 

    """
    Calculate the pressure corresponding to a given energy density. This function uses a piecewise power-law 
    interpolation based on the provided `eps` and `pres` arrays, which represent the energy density and pressure 
    values of the equation of state (EOS) grid, respectively. The `idx` parameter indicates the index in the EOS grid that corresponds to the given energy density.
    Args:        
        epsilon (double): The energy density for which to calculate the pressure, in geometrized units (g/cm^3 converted to g/cm).
        pres (double): An array of pressure values corresponding to the EOS grid, in geometrized units (g/(cm s^2) converted to g/(cm s^2)).
        eps (double): An array of energy density values corresponding to the EOS grid, in geometrized units (g/cm^3 converted to g/cm).
        idx (int): The index in the EOS grid that corresponds to the given energy density. This index is obtained using a binary search on the `eps` array.

    Returns:
        double: The pressure corresponding to the given energy density, calculated using piecewise power-law interpolation
    """

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

    """
    Calculate the derivatives of the TOV equations at a given radius `r` and state vector `y`. 
    This function is designed to be used with a numerical ODE solver, such as those provided by the 
    GNU Scientific Library (GSL). The function computes the derivatives of the pressure, mass, and metric
     functions based on the current state of the system and the equation of state (EOS) parameters.

     Args:
        r (double): The radial coordinate at which to evaluate the derivatives, in geometrized units (cm).
        y (double[]): The state vector at radius `r`, containing the following components:
                - y[0]: Pressure (P) in geometrized units (g/(cm s^2) converted to g/(cm s^2))
                - y[1]: Mass enclosed within radius `r` (m) in geometrized units (g converted to cm)
                - y[2]: Metric function h(r) related to the radial component of the metric
                - y[3]: Metric function b(r) related to the time component of the metric
                - y[4]: Metric function alpha(r) related to the time component of the metric
        f (double[]): An array to store the computed derivatives, where:
                - f[0]: dP/dr, the derivative of pressure with respect to radius
                - f[1]: dm/dr, the derivative of mass with respect to radius
                - f[2]: dh/dr, the derivative of metric function h with respect to radius
                - f[3]: db/dr, the derivative of metric function b with respect to radius
                - f[4]: dalpha/dr, the derivative of metric function alpha with respect to radius
        par (void*): A pointer to additional parameters needed for the calculation. This should point to an array containing:
                - par[0]: A pointer to an array of energy density values corresponding to the EOS grid, in geometrized units.
                - par[1]: A pointer to an array of pressure values corresponding to the EOS grid, in geometrized units.
                - par[2]: A pointer to a double containing the number of points in the EOS grid.

        Returns:
            int: A status code indicating the success of the computation. Returns `GSL_SUCCESS` if the derivatives were computed successfully.
    """

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
            rhocent (float): The central energy density of the star in geometrized units (g/cm^3 converted to g/cm).
            pcent (float): The central pressure of the star in geometrized units (g/(cm s^2) converted to g/(cm s^2)).
            adindcent (float, optional): The adiabatic index at the center of the star. Default is 2, which 
            corresponds to a relativistic degenerate gas.

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


    """    
    
    Solve the TOV equations for a given central energy density and equation of state (EOS) defined by
     `eos_eps` and `eos_pres`. The function integrates the TOV equations from the center of the star 
     (where the radius is small) outward until the pressure drops below a specified minimum value. The function returns the mass, radius, tidal deformability, and metric function values at each radius.

     Args:
        rhocent (float): The central energy density of the star in cgs units (g/cm^3).
        eos_eps (np.ndarray): An array of energy density values for the EOS grid in cgs units (g/cm^3).
        eos_pres (np.ndarray): An array of pressure values for the EOS grid in cgs units (g/(cm s^2)).
        atol (float): The absolute tolerance for the ODE solver.
        rtol (float): The relative tolerance for the ODE solver.
        hmax (float): The maximum step size for the ODE solver in cm.
        step (float): The initial step size for the ODE solver in cm.


    Returns:        tuple: A tuple containing the following elements:
            - Mb (float): The mass of the neutron star in grams.
            - Rns (float): The radius of the neutron star in cm.
            - tidal (float): The tidal deformability of the neutron star (dimensionless).
            - Gtt (np.ndarray): A 2D array containing the radius and the metric function values at each radius. 
            The first column contains the radius in cm, and the second column contains the metric function values 
            (g_tt) in geometrized units.

    """
            

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

