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


# in terms of r
cdef double EofP(double pressure, double eps[], double pres[], int idx) nogil:
    if idx == 0:
        eds = eps[0] * pow(pressure / pres[0], 3. / 5.)
    if idx > 0:
        ci = log(pres[idx] / pres[idx - 1]) / log(eps[idx] / eps[idx - 1])
        eds = eps[idx - 1] * pow(pressure / pres[idx - 1], 1. / ci)
    return eds

cdef double PofE(double epsilon, double pres[], double eps[], int idx) nogil:
    if idx==0:
        pressure = pres[0]*pow(epsilon/eps[0], 5./3.)
    if idx>0.:
        ci = log(pres[idx]/pres[idx-1])/log(eps[idx]/eps[idx-1])
        pressure = pres[idx-1] * pow(epsilon/eps[idx-1], ci)
    return pressure



cdef int TOV_single(double r, const double y[], double f[], void * par) noexcept nogil:

    cdef double p
    cdef double eps

    cdef double *rhotest = (<double**> par)[0]
    cdef double *prestest = (<double**> par)[1]
    cdef double *num_double = (<double**>par)[2]
    cdef int num = <int> num_double[0]
    p = sqrt(y[0]*y[0])
    cdef int idx = binarySearch(prestest, 0, num, p)
    eps = EofP(p, rhotest, prestest, idx)


    f[0] = -(eps + p) * (y[1] + 4. * pi * pow(r,3) * p)
    f[0] *= pow(r * (r - 2.*y[1]), -1)


    f[1] = 4.*pi*pow(r,2) * eps


    f[2] = (y[1] + 4.*pi*pow(r,3) * p) * pow(r*(r - 2.*y[1]), -1)

    return GSL_SUCCESS


cdef int TOV_complete(double r, const double y[], double f[],void * par) noexcept nogil:

    cdef double pb
    cdef double epsb

    cdef double pdm
    cdef double epsdm

    cdef double *epsb_test = (<double**> par)[0]
    cdef double *presb_test = (<double**> par)[2]

    cdef double *epsdm_test = (<double**> par)[1]
    cdef double *presdm_test = (<double**> par)[3]

    cdef double *numb_double = (<double**> par)[4]
    cdef int numb = <int> numb_double[0]

    cdef double *numdm_double = (<double**> par)[5]
    cdef int numdm = <int> numdm_double[0]

    pb = sqrt(y[0]*y[0])
    cdef int idx_b = binarySearch(presb_test, 0, numb, pb)
    epsb = EofP(pb, epsb_test, presb_test, idx_b)

    pdm = sqrt(y[1]*y[1])
    cdef int idx_dm = binarySearch(presdm_test, 0, numdm, pdm)
    epsdm = EofP(pdm, epsdm_test, presdm_test, idx_dm)

    cdef double P = pb + pdm
    cdef double Mb = y[2]
    cdef double Mdm = y[3]
    cdef double M = Mb + Mdm

    f[0] = -(epsb + pb) * (M + 4 * pi * pow(r,3) * P) / (pow(r,2)-2 * M * r)
    #f[0]*= pow((pow(r,2)-2 * M * r),-1)
    f[1] = -(epsdm + pdm) * (M + 4 * pi * pow(r,3) * P) / (pow(r,2)-2 * M * r)
    #f[1]*= pow((pow(r,2)-2 * M * r),-1)

    f[2] = 4*pi*pow(r,2)*epsb
    f[3] = 4*pi*pow(r,2)*epsdm

    f[4] = (M + 4 * pi * pow(r,3) * P) / (pow(r,2)-2 * M * r)

    return GSL_SUCCESS

def initial_conditions(double rhobcent,double rhodmcent, double pbcent, double pdmcent):
        """
        Set the initial conditions for solving the structure equations.


        Args:
            rhobcent (double): baryonic central density  in geometrized units.
            rhodmcent (double): ADM central density  in geometrized units.
            pbcent (double): baryonic central pressure  in geometrized units.
            pdmcent (double): ADM central pressure in geometrized units.

            
        Returns:
            tuple: tuple containing:

            
                - **dr** (*float*): The initial stepsize in cm.
                - **intial** (*array*): A np array storing the initial conditions.
        """
        cdef double r = 4.441e-16
        cdef double dr = 10.

        cdef double Pb0, Pdm0, mb0, mdm0

        Pb0 = pbcent - (2.*pi/3.)*(pbcent + rhobcent) *(3.*pbcent + rhobcent)*r**2.
        Pdm0 = pdmcent - (2.*pi/3.)*(pdmcent + rhodmcent) *(3.*pdmcent + rhodmcent)*r**2.
        mb0 =  4./3. *pi *rhobcent*r**3.
        mdm0 = 4./3. *pi *rhodmcent*r**3.

        initial = np.array([Pb0, Pdm0, mb0, mdm0, 0.0])

        return dr, initial

def solveTOVdm(double rhobcent, double rhodmcent, eos_epsb, eos_presb, eos_epsdm, eos_presdm, dm_halo, two_fluid_tidal,double atol,
              double rtol, double hmax, double step):

    #cdef double atol=1e-6
    #cdef double rtol=1e-4
    #cdef double hmax=1000.
    #cdef double step= 0.46

    cdef int i
    cdef double Pmin = 1e4 * G * pow(c,-4)

    

    # Making sure there are no double values in baryonic pressure and energy density arrays
    eos_presb, indices = np.unique(np.log10(eos_presb).round(decimals=3),
                                  return_index=True)
    eos_presb = 10**eos_presb * G * pow(c,-4) #scaled to geometrized
    eos_epsb = eos_epsb[np.sort(indices)] * G * pow(c,-2) #scaled to geometrized

    cdef int numb = len(eos_presb)
    cdef double numb_double = float(numb)

    # Making sure there are no double values in dark matter pressure and energy density arrays
    eos_presdm, indices = np.unique(np.log10(eos_presdm).round(decimals=3),
                                  return_index=True)

    eos_presdm = 10**(eos_presdm) * G * pow(c,-4) #scaled to geometrized
    eos_epsdm = eos_epsdm[np.sort(indices)] * G * pow(c,-2) #scaled to geometrized

    cdef int numdm = len(eos_presdm)
    cdef double numdm_double = float(numdm)

    # create C array from the python array
    cdef np.ndarray[double, ndim=1, mode="c"] epsb_cython = np.asarray(eos_epsb, dtype='float', order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] presb_cython = np.asarray(eos_presb, dtype='float', order="C")

    cdef np.ndarray[double, ndim=1, mode="c"] epsdm_cython = np.asarray(eos_epsdm, dtype='float', order="C")
    cdef np.ndarray[double, ndim=1, mode="c"] presdm_cython = np.asarray(eos_presdm, dtype='float', order="C")

    # Set initial conditions for solving the TOV equations
    rhobcent = rhobcent * G * pow(c,-2)
    rhodmcent = rhodmcent * G * pow(c,-2)

    cdef int idx = binarySearch(&epsb_cython[0], 0, numb, rhobcent)
    cdef double pbcent = PofE(rhobcent, &presb_cython[0], &epsb_cython[0], idx)

    cdef int idxdm = binarySearch(&epsdm_cython[0], 0, numdm, rhodmcent)
    cdef double pdmcent = PofE(rhodmcent, &presdm_cython[0], &epsdm_cython[0], idxdm)

    cdef double r = 4.441e-16

    dr, initial = initial_conditions(rhobcent,rhodmcent, pbcent,pdmcent)
    cdef double *stateTOV = <double*> malloc(5 * sizeof(double))
    for i in range(5):
        stateTOV[i] = initial[i]

    # Defining extra arguments for the TOV solver function
    cdef double *params[6]
    params[0] = &epsb_cython[0]
    params[1] = &epsdm_cython[0]
    params[2] = &presb_cython[0]
    params[3] = &presdm_cython[0]
    params[4] = &numb_double
    params[5] = &numdm_double

    # Initialize the ODE integrator
    cdef int dim = 5


    #Could be the source of the "integration limit and/or step direction not consistent" problem
    cdef gsl_odeiv2_system sys = [TOV_complete, NULL, dim, &params]
    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, r, atol, rtol)

    cdef double epsb, epsdm, dpbdr,dpdmdr, dmbdr, dmdmdr
    cdef double r_c = r


    Array = np.empty((0, 6), float)
    Array = np.append(Array,np.array([[r,initial[2]+initial[3],rhobcent,pbcent,rhodmcent,pdmcent]]),axis = 0)

    # Define an array to store the gtt component of the metric in
    #Gtt = np.empty((0, 2), float)
    dalphadr = 0.0
    #Gtt = np.append(Gtt, np.array([[r, dalphadr]]), axis=0)

    ## Integrate the TOV equations ##
    ## stateTOV[0] = pb, stateTOV[1] = pdm, stateTOV[2] = mb, stateTOV[3] = mdm, stateTOV[4] = alpha ##
    while (stateTOV[0]>Pmin and stateTOV[1]>Pmin and dr-r>1e-14):

        status = gsl_odeiv2_driver_apply(d, &r, dr, stateTOV)
        if (status != GSL_SUCCESS):
            break

        stateTOV[0] = sqrt(pow(stateTOV[0], 2))
        stateTOV[1] = sqrt(pow(stateTOV[1], 2))

        pb = stateTOV[0]
        pdm = stateTOV[1]
        
        idx = binarySearch(&presb_cython[0], 0, numb, stateTOV[0])
        epsb = EofP(stateTOV[0], &epsb_cython[0], &presb_cython[0], idx)

        idxdm = binarySearch(&presdm_cython[0], 0, numdm, stateTOV[1])
        epsdm = EofP(stateTOV[1], &epsdm_cython[0], &presdm_cython[0], idxdm)

        dpbdr = -(epsb + stateTOV[0]) * (stateTOV[2]+stateTOV[3] + 4.*pi*pow(r,3) * (stateTOV[0]+stateTOV[1]))
        dpbdr = dpbdr*pow(r*(r - 2.*(stateTOV[2]+stateTOV[3])), -1)

        dpdmdr = -(epsdm + stateTOV[1]) * (stateTOV[2]+stateTOV[3] + 4.*pi*pow(r,3) * (stateTOV[0]+stateTOV[1]))
        dpdmdr = dpdmdr*pow(r*(r - 2.*(stateTOV[2]+stateTOV[3])), -1)

        dalphadr = (stateTOV[2]+stateTOV[3] + 4.*pi*pow(r,3) * (stateTOV[0]+stateTOV[1]))
        dalphadr = dalphadr*pow(r*(r - 2.*(stateTOV[2]+stateTOV[3])), -1)

        Array = np.append(Array, np.array([[r,stateTOV[2]+stateTOV[3],epsb,pb,epsdm,pdm]]), axis=0) #added 10/24

        #Gtt = np.append(Gtt, np.array([[r, stateTOV[4]]]), axis=0)


        dmbdr = 4.*pi*pow(r,2) * epsb
        dmdmdr = 4.*pi*pow(r,2) * epsdm
        tmp1 = pow(pow(stateTOV[2],-1) * dmbdr - pow(stateTOV[0],-1)*dpbdr, -1)
        tmp2 = pow(pow(stateTOV[3],-1) * dmdmdr - pow(stateTOV[1],-1)*dpdmdr, -1)
        tmp3 = min(tmp1, tmp2)
        dr = r + step * tmp3

        if dr-r == 0:
            break


    gsl_odeiv2_driver_free (d)
    cdef double *params_single[3]
    cdef int dim_single = 3
    cdef gsl_odeiv2_system sys_single = [TOV_single, NULL, dim_single, &params_single]
    cdef gsl_odeiv2_driver * d_single = gsl_odeiv2_driver_alloc_y_new(&sys_single, gsl_odeiv2_step_rk8pd, r, atol, rtol)
    cdef double *stateTOV_single = <double*> malloc(3 * sizeof(double))

    cdef double Mb, Rns, Mdm_core, Mdm_halo, Rdm_core, Rdm_halo

    # if Pb > Pdm at the radius in which the while loop breaks -> DM core
    if(stateTOV[0] > stateTOV[1]): 
        # print('dark matter core')
        params_single[0] = &epsb_cython[0] 
        params_single[1] = &presb_cython[0] 
        params_single[2] = &numb_double 

        stateTOV_single[0] = stateTOV[0]# Baryonic pressure
        stateTOV_single[1] = stateTOV[2]+stateTOV[3] # Mb + Mdm
        stateTOV_single[2] = stateTOV[4] # alpha

        Rdm_core = r
        timeout = time.time() + 2 #2 second timeout to avoid any lack of convergence [Seems to happen when ADM core pushed baryonic mass near minumum mass. Loop typically takes 0.03-0.9 seconds
        # May need to only include the time statement in an if statement? Will need to check tonight!
        while (stateTOV_single[0]>Pmin and time.time() <= timeout):
            status = gsl_odeiv2_driver_apply(d_single, &r, dr, stateTOV_single)

            if (status != GSL_SUCCESS):
                #printf ("error, return value=%d\n", status)
                break


            #in the geometrized units, the central pressures are usually of the order 10^1, at the highest! However, sometimes the pressure skips to infinity for extremely high ADM central pressures
            if(stateTOV_single[0] >= 1e30):
                break

            stateTOV_single[0] = sqrt(pow(stateTOV_single[0], 2))
            idx = binarySearch(&presb_cython[0], 0, numb, stateTOV_single[0])
            eps = EofP(stateTOV_single[0], &epsb_cython[0], &presb_cython[0], idx)
            dpdr = -(eps + stateTOV_single[0]) * (stateTOV_single[1] + 4.*pi*pow(r,3) * stateTOV_single[0])
            dpdr = dpdr * pow(r*(r - 2. * stateTOV_single[1]), -1)
            dmdr = 4.*pi*pow(r,2) * eps

            Array = np.append(Array, np.array([[r,stateTOV_single[1],eps,stateTOV_single[0],0,0]]), axis=0) #added 10/24

            #Gtt = np.append(Gtt, np.array([[r, stateTOV_single[2]]]), axis=0)


            dr = r + step * pow(pow(stateTOV_single[1],-1) * dmdr - pow(stateTOV_single[0],-1)*dpdr, -1)

        Rns = r
        Mdm_core = stateTOV[3]
        Mdm_halo = 0.0
        Rdm_halo = 0.0
        Mb = stateTOV_single[1]-Mdm_core






    # if Pdm > Pb at the radius in which the while loop breaks -> DM halo
    if(stateTOV[1] > stateTOV[0]): 
        # print('dark matter halo')
        if dm_halo == False:
            Mdm_halo = 0.0
            Mdm_core = stateTOV[3]
            Rdm_halo = 999.0*pow(10,5)
            Rdm_core = 0.0
            Rns = 0.0
            Mb = stateTOV[2]

        else:
            params_single[0] = &epsdm_cython[0] 
            params_single[1] = &presdm_cython[0]
            params_single[2] = &numdm_double

            stateTOV_single[0] = stateTOV[1] # DM pressure
            stateTOV_single[1] = stateTOV[2]+stateTOV[3] # Mb + Mdm
            stateTOV_single[2] = stateTOV[4] # alpha

            Rns = r
            Rdm_core = r
            Mb = stateTOV[2]
            Mdm_core = stateTOV[3]

            while (stateTOV_single[0]>Pmin and dr-r>1e-14): ##with halos that are very large it seems setting >0 is the way to go as opposed to using Pmin##
                status = gsl_odeiv2_driver_apply(d_single, &r, dr, stateTOV_single)
                if (status != GSL_SUCCESS):
                    printf ("error, return value=%d\n", status)
                    break

                stateTOV_single[0] = sqrt(pow(stateTOV_single[0], 2))

                idx = binarySearch(&presdm_cython[0], 0, numdm, stateTOV_single[0])
                eps = EofP(stateTOV_single[0], &epsdm_cython[0], &presdm_cython[0], idx)
                dpdr = -(eps + stateTOV_single[0]) * (stateTOV_single[1] + 4.*pi*pow(r,3) * stateTOV_single[0])
                dpdr = dpdr * pow(r*(r - 2. * stateTOV_single[1]), -1)
                dmdr = 4.*pi*pow(r,2) * eps
                #Gtt = np.append(Gtt, np.array([[r, stateTOV_single[2]]]), axis=0)
                dr = r + step * pow(pow(stateTOV_single[1],-1) * dmdr - pow(stateTOV_single[0],-1)*dpdr, -1)
                Array = np.append(Array, np.array([[r,stateTOV_single[1],0,0, eps,stateTOV_single[0]]]), axis=0) #added 10/24

            
            Mdm_halo = stateTOV_single[1]-Mdm_core-Mb
            Rdm_halo = r



    gsl_odeiv2_driver_free (d_single)
    free(stateTOV)
    free(stateTOV_single)

    Mb = Mb * pow(c,2) / G #scaled back into cgs units of grams
    Mdm_core = Mdm_core * pow(c,2) / G #scaled back into cgs units of grams
    Mdm_halo = Mdm_halo * pow(c,2) / G #scaled back into cgs units of grams

    cdef double tidal
    if two_fluid_tidal == False:
        tidal = 0
    else:
        from neost.tovsolvers.TidalDef import solveTidal
        tidal  = solveTidal(Array,dm_halo)

    #if Rdm_halo > Rns:
        #Gtt[:,1] = Gtt[:,1] - (stateTOV_single[2] - 0.5*log(1 - 2 * (Mb + Mdm_core+Mdm_halo)/Rdm_halo))
    #else:
        #Gtt[:,1] = Gtt[:,1] - (stateTOV_single[2] - 0.5*log(1 - 2 * (Mb + Mdm_core)/Rns))
    return Mb, Rns, Mdm_core, Mdm_halo, Rdm_core, Rdm_halo,tidal