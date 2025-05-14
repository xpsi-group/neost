# Standard libraries
import pathlib

# 3rd party
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpi4py import MPI
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
import corner

# Local imports
import neost
from neost.eos import polytropes, tabulated
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
import neost.global_imports as global_imports

# Constants
c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons

# Define color scheme
colors = np.array(["#c6878f", "#b79d94", "#969696", "#67697c", "#233b57", "#BCBEC7"])

# Units
dyncm2_to_MeVfm3 = 1./(1.6022e33)
gcm3_to_MeVfm3 = 1./(1.7827e12)
oneoverfm_MeV = 197.33


def m1(mc, m2):
    num1 = (2./3.)**(1./3.)*mc**5.
    denom1 = (9*m2**7. *mc**5. + np.sqrt(3.)*np.sqrt(abs(27*m2**14.*mc**10.-4.*m2**9.*mc**15.)))**(1./3.)
    denom2 = 2.**(1./3.)*3.**(2./3.)*m2**3.
    return num1/denom1 + denom1/denom2

def calc_bands(x, y):
    miny = np.zeros((len(y),3))
    maxy = np.zeros((len(y),3))

    for i in range(len(y)):
        z = y[i][y[i]>0.0]
        if len(z)<200:
            print('sample too small for %.2f' %x[i])
            continue
        kde = gaussian_kde(z)
        testz = np.linspace(min(z),max(z), 1000)
        pdf = kde.pdf(testz)
        array = pdf
        index_68 = np.where(np.cumsum(np.sort(array)[::-1]) < sum(array)*0.6827)[0]
        index_68 = np.argsort(array)[::-1][index_68]
        index_95 = np.where(np.cumsum(np.sort(array)[::-1]) < sum(array)*0.95)[0]
        index_95 = np.argsort(array)[::-1][index_95]
        miny[i] =  x[i], min(testz[index_68]), min(testz[index_95])
        maxy[i] =  x[i], max(testz[index_68]), max(testz[index_95])

    miny = miny[~np.all(miny == 0, axis=1)]
    maxy = maxy[~np.all(maxy == 0, axis=1)]
    return miny, maxy

def get_quantiles(array, quantiles=[0.025, 0.5, 0.975]):
        contours = np.nanquantile(array, quantiles)
        low = contours[0]
        median = contours[1]
        high = contours[2]
        minus = low - median
        plus = high - median
        return np.round(median,2),np.round(plus,2),np.round(minus,2)


def compute_table_data(root_name, EOS, variable_params, static_params,dm = False):
    """
    Function to compute the table data in Raaijmakers et al. 2021 & Rutherford et al. 2024.
    In particular: M_TOV, R_TOV, eps_cent_TOV, rho_cent_TOV, P_cent_TOV, R_1.4, eps_cent_1.4, rho_cent_1.4, P_cent_1.4,
    R_2.0, eps_cent_2.0, rho_cent_2.0, P_cent_2.0. Begining from v2.0.0 onward, the user has the choice to include ADM or not.


    Parameters
    ----------

    root_name: str
        Name of the inference run to refer back to. Used to get the Multinest outputs.

    EOS: obj
        equation of state object initialized in the inference script, i.e., the parameters that are sampled during inferencing.

    variable_params: dict
        Variable parameters in the inference script.

    static_params: dict
        Static parameters in the inference script, i.e., the parameters that are held static during inference sampling.

    dm: bool
        If True ADM is included when computing the table data.


    """
    ewposterior = np.loadtxt(root_name + 'post_equal_weights.dat')
    num_samples = num_samples
    print("Total number of samples is %d" %(num_samples))
    try:
        Data_array = np.loadtxt(root_name + 'table_data.txt')
        print('M_TOV: ', get_quantiles(Data_array[:,0]))
        print('R_TOV: ', get_quantiles(Data_array[:,1]))
        print('eps_cent TOV: ', get_quantiles(Data_array[:,2]))
        print('rho_cent TOV: ', get_quantiles(Data_array[:,3]))
        print('P_cent TOV: ', get_quantiles(Data_array[:,4]))
        print('R_1.4: ', get_quantiles(Data_array[:,5]))
        print('eps_cent 1.4: ', get_quantiles(Data_array[:,6]))
        print('rho_cent 1.4: ', get_quantiles(Data_array[:,7]))
        print('P_cent 1.4: ', get_quantiles(Data_array[:,8]))
        print('R_2.0: ', get_quantiles(Data_array[:,9]))
        print('eps_cent 2.0: ', get_quantiles(Data_array[:,10]))
        print('rho_cent 2.0: ', get_quantiles(Data_array[:,11]))
        print('P_cent 2.0: ', get_quantiles(Data_array[:,12]))
        print('Delta R = R_2.0 - R_1.4: ', get_quantiles(Data_array[:,9] - Data_array[:,5]))
    except OSError:
        Data_array = np.zeros((num_samples,13)) #contains Mtov, Rtov, eps_cent TOV, rho_cent TOV, P_cent TOV,R 1.4, eps_cent 1.4, rho_cent 1.4,
                                                                #P_cent 1.4, R 2.0, eps_cent 2.0, rho_cent 2.0, P_cent 2.0
                                                                #NOTE: ALL VALUES ARE THEIR ADMIXED VERSIONS WHEN dm == True!!



        for i in range(0, num_samples, 1):
            pr = ewposterior[i][0:len(variable_params)]
            par = {e:pr[j] for j, e in enumerate(list(variable_params.keys()))}
            par.update(static_params)
            EOS.update(par, max_edsc=True)

            edsrho = UnivariateSpline(EOS.energydensities, EOS.massdensities, k=1, s=0)
            eps = np.logspace(14.4, np.log10(EOS.max_edsc), 40)
            M = np.zeros(len(eps))
            R = np.zeros(len(eps))

            if dm == False:
                max_rhoc = edsrho(EOS.max_edsc) / rho_ns #division by rho_ns gives max_rhoc in terms of n_c/n_0 as mass density and number density only differ by a factor the mass of baryon, which is canceled out in this fraction
                for j, e in enumerate(eps):
                    star = Star(e)
                    star.solve_structure(EOS.energydensities, EOS.pressures)
                    M[j] = star.Mrot
                    R[j] = star.Req

                M, indices = np.unique(M, return_index=True)
                MR = UnivariateSpline(M, R[indices], k=1, s=0, ext=1)
                epsM = UnivariateSpline(M, eps[indices], k=1, s=0,ext = 1)

                R_14 = MR(1.4)
                if R_14 == 0:
                    R_14 = np.nan # set to be nan so they don't impact the quantiles b/c we are using np.nanquantiles
                    eps_14 = np.nan
                    rho_14 = np.nan
                    pres_14 = np.nan
                else:
                    eps_14 = epsM(1.4)
                    rho_14 = edsrho(eps_14) / rho_ns + edsrhodm()
                    pres_14 = EOS.eos(eps_14)

                R_2 = MR(2.0)
                if R_2 == 0:
                    R_2 = np.nan # see above for reasoning
                    eps_2 = np.nan
                    rho_2 = np.nan
                    pres_2 = np.nan # see above for reasoning
                else:
                    eps_2 = epsM(2.0)
                    rho_2 = edsrho(eps_2) / rho_ns
                    pres_2 = EOS.eos(eps_2)

                row = [EOS.max_M, EOS.Radius_max_M, np.log10(EOS.max_edsc), max_rhoc, np.log10(EOS.eos(EOS.max_edsc)),R_14, np.log10(eps_14), rho_14, np.log10(pres_14),R_2, np.log10(eps_2), rho_2, np.log10(pres_2)]

            else:
                edsrho_dm = UnivariateSpline(EOS.energydensities_dm, EOS.massdensities_dm, k=1, s=0, ext = 1)

                epsdm_max = EOS.find_epsdm_cent(EOS.adm_fraction, EOS.max_edsc)
                max_rhocdm = edsrho_dm(epsdm_max) / rho_ns
                max_rhocb = edsrho(EOS.max_edsc) / rho_ns
                max_rhoc = max_rhocb + max_rhocdm

                epsdm = np.zeros(len(eps))
                Rdm = np.zeros(len(eps))
                Mdm = np.zeros(len(eps))


                for j, e in enumerate(eps):
                    epsdm_cent = EOS.find_epsdm_cent(EOS.adm_fraction,eps)
                    epsdm[i] = epsdm_cent
                    star = Star(e,epsdm_cent)
                    star.solve_structure(EOS.energydensities, EOS.pressures,EOS.energydensities_dm, EOS.pressures_dm, EOS.dm_halo) #EOS.two_fluid_tidal not needed in this section since only MR
                                                                                                                                    #so default value is used (False).
                    M[j] = star.Mrot
                    R[j] = star.Req
                    Rdm[j] = star.Rdm
                    Mdm[j] = star.Mdm

                M, indices = np.unique(M, return_index=True)
                index_max_M = np.argmax(M[indices])
                Radius_max_M = R[index_max_M]


                MR = UnivariateSpline(M, R[indices], k=1, s=0, ext=1)
                epsdm_Mdm = UnivariateSpline(Mdm[indicies], epsdm[indices], k=1, s=0, ext=1)

                eps_total = eps + epsdm
                epsM = UnivariateSpline(M, eps_total[indices], k=1, s=0,ext = 1)

                eos_dm = UnivariateSpline(EOS.energydensities_dm, EOS.pressures_dm, k=1, s=0,ext = 1)

                R_14 = MR(1.4)
                if R_14 == 0:
                    R_14 = np.nan # set to be nan so they don't impact the quantiles b/c we are using np.nanquantiles
                    eps_14 = np.nan
                    rho_14 = np.nan
                    pres_14 = np.nan
                else:
                    eps_14 = epsM(1.4)
                    M_chi = EOS.adm_fraction/100*1.4 #F_chi = M_chi/M_total*100
                    epsdm_14 = epsdm_Mdm(M_chi)
                    epsb_14 = eps_14 - epsdm_14

                    rho_14 = edsrho(epsb_14) / rho_ns + edsrho_dm(epsdm_14) / rho_ns

                    pres_14 = EOS.eos(epsb_14) + eos_dm(epsdm_14)

                R_2 = MR(2.0)
                if R_2 == 0:
                    R_2 = np.nan # see above for reasoning
                    eps_2 = np.nan
                    rho_2 = np.nan
                    pres_2 = np.nan # see above for reasoning
                else:
                    M_chi = EOS.adm_fraction/100*2.0 #F_chi = M_chi/M_total*100 ---> M_chi = F_chi/100*M_total
                    epsdm_2 = epsdm_Mdm(M_chi)
                    epsb_2 = eps_2 - epsdm_2

                    rho_2 = edsrho(epsb_2) / rho_ns + edsrho_dm(epsdm_2) / rho_ns

                    pres_2 = EOS.eos(epsb_2) + eos_dm(epsdm_2)


                row = [max(M), Radius_max_M, np.log10(EOS.max_edsc + epsdm_max), max_rhoc, np.log10(EOS.eos(EOS.max_edsc) + eos_dm(epsdm_max)),R_14, np.log10(eps_14), rho_14, np.log10(pres_14),R_2, np.log10(eps_2), rho_2, np.log10(pres_2)]


            for k in range(len(row)):
                # Some of the values in row may be arrays of shape (1,),
                # which causes the line "Data_array[i] = row" to fail for np > 1.23.5.
                # Some values are ndarrays with shape () which is fine, so check for that.
                # If the value is an array and doesn't have the shape (),
                # then check that its shape is indeed (1,) and extract the value.
                if hasattr(row[k], "shape") and row[k].shape != ():
                    assert(row[k].shape == (1,))
                    row[k] = row[k][0]
            Data_array[i,:] = row
        # save everything
        np.savetxt(root_name + 'table_data.txt', Data_array)
        print('M_TOV: ', get_quantiles(Data_array[:,0]))
        print('R_TOV: ', get_quantiles(Data_array[:,1]))
        print('eps_cent TOV: ', get_quantiles(Data_array[:,2]))
        print('rho_cent TOV: ', get_quantiles(Data_array[:,3]))
        print('P_cent TOV: ', get_quantiles(Data_array[:,4]))
        print('R_1.4: ', get_quantiles(Data_array[:,5]))
        print('eps_cent 1.4: ', get_quantiles(Data_array[:,6]))
        print('rho_cent 1.4: ', get_quantiles(Data_array[:,7]))
        print('P_cent 1.4: ', get_quantiles(Data_array[:,8]))
        print('R_2.0: ', get_quantiles(Data_array[:,9]))
        print('eps_cent 2.0: ', get_quantiles(Data_array[:,10]))
        print('rho_cent 2.0: ', get_quantiles(Data_array[:,11]))
        print('P_cent 2.0: ', get_quantiles(Data_array[:,12]))
        print('Delta R = R_2.0 - R_1.4: ', get_quantiles(Data_array[:,9] - Data_array[:,5]))

def load_equal_weighted_samples(path, sampler, identifier):
    # For Multinest, 'path' is the directory containing all the output files.
    # For Ultranest, it's the base directory of the output files (i.e. 'path' should contain a subdirectory called 'chains'
    assert(sampler.lower() in ['ultranest', 'multinest'])
    if sampler == 'ultranest' and identifier != '':
        warnings.warn(f'Ultranest does not support run names, ignoring identifier {identifier}')
    # Assume Ultranest is used
    sample_file = f'{path}/chains/equal_weighted_post.txt'
    skiprows = 1 # This is to ignore the first line, which contains the parameter names
    if sampler.lower() == 'multinest':
        skiprows = 0 # No need to skip any lines with Multinest
        sample_file = f'{path}/{identifier}post_equal_weights.dat'

    # Load and return the samples
    print(f'Analyzing file {sample_file}')
    return np.loadtxt(sample_file, skiprows=skiprows)

def save_auxiliary_data(path, identifier, data, fnames):
    # data and fnames should be lists.
    assert(len(data) == len(fnames))
    for i in range(len(data)):
        extension = pathlib.Path(fnames[i]).suffix.lower()
        savefunc = np.savetxt if extension == '.txt' else np.save
        fname = f'{path}/{identifier}{fnames[i]}'
        savefunc(fname, data[i])
        print(f'Writing {fname} to disk')

def print_samples_per_core(samples):
    print('Number of samples to be analyzed per core:', end=' ')
    for a in samples:
        print(len(a), end=' ')
    print()

def compute_auxiliary_data(path, EOS, variable_params, static_params, chirp_masses, dm=False, sampler='multinest', identifier=''):
    """
    Function to compute the posterior auxiliary data used to generate standard NEoST plots, such as, the pressures, (if dm = True)
    the baryonic pressure, mass-radius posteriors, and p-eps posteriors.


    Parameters
    ----------

    root_name: str
        Name of the inference run to refer back to. Used to get the Multinest outputs.

    EOS: obj
        equation of state object initialized in the inference script, i.e., the parameters that are sampled during inferencing.

    variable_params: dict
        Variable parameters in the inference script.

    static_params: dict
        Static parameters in the inference script, i.e., the parameters that are held static during inference sampling.

    chirp_masses: list
        List determining if GW data is included. If None, just MR only.

    dm: bool
        If True, ADM is included when computing the table data.


    """
    # Set up some MPI things
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank() # The rank of the current MPI process
    num_processes = comm.Get_size() # Number of MPI processes
    samples = [[] for i in range(num_processes)] # This is essentially a rearranged 'equal_weighted_samples'

    # Load samples
    equal_weighted_samples = load_equal_weighted_samples(path, sampler, identifier)
    num_samples = len(equal_weighted_samples)
    print(f'Total number of samples is {num_samples}')

    # Get number of stars and set 'flag' that avoids unneeded calculations
    num_stars = len(np.array([v for k,v in variable_params.items() if 'rhoc' in k]))
    flag = True if len(list(variable_params.keys())) == num_stars else False

    # Grids
    # More points are added to account for larger energy density spread from ADM
    # total ADM [1e12,1e18] + baryonic energy densities [1e14.2,1e16]
    num_grid_points = 200 if dm else 50
    masses = np.linspace(.2, 2.9, num_grid_points)
    num_masses = num_grid_points # TODO maybe this isn't necessary?
    energydensities = np.logspace(14.2, 16, num_grid_points)
    num_energydensities = num_grid_points # TODO maybe this isn't necessary?
    mass_radius = np.zeros((num_samples, 2))
    radii = np.zeros((num_masses, num_samples))
    pressures = np.zeros((num_masses, num_samples))
    pressures_rho = np.zeros((num_masses, num_samples))
    scattered = []

    if dm:
        # We can always specify these even if they're not used I think
        energydensities_b = np.logspace(14.2, 16, 200)
        energydensities_dm = np.logspace(12, 18, 200)
        energydensities = energydensities_b + energydensities_dm # Overwrite energydensities, length is the same

        pressures_dm = np.zeros((num_energydensities, num_samples))
        pressures_rho_dm = np.zeros((num_energydensities, num_samples))
        pressures_b = np.zeros((num_energydensities, num_samples))
        pressures_rho_b = np.zeros((num_energydensities, num_samples))

    if not flag:
        # We can always specify these even if they're not used I think
        minradii = np.zeros((3, num_masses))
        maxradii = np.zeros((3, num_masses))
        minpres = np.zeros((3, num_energydensities))
        maxpres = np.zeros((3, num_energydensities))
        minpres_rho = np.zeros((3, num_energydensities))
        maxpres_rho = np.zeros((3, num_energydensities))

    for i in range(0, num_samples, 1):

        pr = ewposterior[i][0:len(variable_params)]
        par = {e:pr[j] for j, e in enumerate(list(variable_params.keys()))}
        par.update(static_params)
        EOS.update(par, max_edsc=True)

        rhocs = np.logspace(14.5, np.log10(EOS.max_edsc), 30)
        rhocsdm = np.zeros_like(rhocs)

        M = np.zeros(len(rhocs))
        R = np.zeros(len(rhocs))

        
        rhocpar = np.array([10**v for k,v in par.items() if 'rhoc' in k])
        tmp = []

        if dm == False:
            rhopres = UnivariateSpline(EOS.massdensities, EOS.pressures, k=1, s=0)
            edsrho = UnivariateSpline(EOS.energydensities, EOS.massdensities, k=1, s=0)
            max_rhoc = edsrho(EOS.max_edsc)
            pressures_rho[:,i][energydensities<max_rhoc] = rhopres(energydensities[energydensities<max_rhoc])
            pressures[:,i][energydensities<EOS.max_edsc] = EOS.eos(energydensities[energydensities<EOS.max_edsc])

            for j, e in enumerate(rhocs):
                star = Star(e)
                star.solve_structure(EOS.energydensities, EOS.pressures)
                M[j] = star.Mrot
                R[j] = star.Req


            M, indices = np.unique(M, return_index=True)
            MR = UnivariateSpline(M, R[indices], k=1, s=0, ext=1)
            rhocM = UnivariateSpline(M, rhocs[indices], k=1, s=0, ext = 1)

            for j, e in enumerate(rhocpar):
                star = Star(e)
                star.solve_structure(EOS.energydensities, EOS.pressures)
                tmp.append([e, EOS.eos(e), star.Mrot, star.Req, star.tidal])

                if chirp_masses[j] is not None:
                    M2 = m1(chirp_masses[j], tmp[j][2])
                    rhoc = rhocM(M2)
                    star = Star(rhoc)
                    star.solve_structure(EOS.energydensities, EOS.pressures)
                    tmp.append([rhoc, EOS.eos(rhoc), star.Mrot, star.Req, star.tidal])

            scattered.append(tmp)
            radii[:,i] = MR(masses)
            rhoc = np.random.rand() *(np.log10(EOS.max_edsc) - 14.6) + 14.6
            star = Star(10**rhoc)
            star.solve_structure(EOS.energydensities, EOS.pressures)
            mass_radius[i] = star.Mrot, star.Req

        else:
            rhopres = UnivariateSpline(EOS.massdensities, EOS.pressures, k=1, s=0, ext = 1)
            edsrho = UnivariateSpline(EOS.energydensities, EOS.massdensities, k=1, s=0, ext = 1)
            max_rhoc = edsrho(EOS.max_edsc)

            rhopres_dm = UnivariateSpline(EOS.massdensities_dm, EOS.pressures_dm, k=1, s=0, ext = 1)
            edsrho_dm = UnivariateSpline(EOS.energydensities_dm, EOS.massdensities_dm, k=1, s=0, ext = 1)
            eos_dm = UnivariateSpline(EOS.energydensities_dm, EOS.pressures_dm, k=1, s=0, ext = 1)

            max_edsc_dm = EOS.find_epsdm_cent(EOS.adm_fraction,EOS.max_edsc)

            if EOS.reach_fraction == False:
                max_rhoc_dm = 0.0
                max_edsc_dm = 0.0
            else:
                max_rhoc_dm = edsrho_dm(max_edsc_dm)


            max_rhoc_admixed = max_rhoc + max_rhoc_dm
            max_edsc_admixed = EOS.max_edsc + max_edsc_dm

            pressures_rho_b[:,i][energydensities_b<max_rhoc] = rhopres(energydensities_b[energydensities_b<max_rhoc])

            dm_pres_rho = rhopres_dm(energydensities_dm[energydensities_dm<max_rhoc_dm])

            if len(dm_pres_rho) != 0:
                pressures_rho_dm[:,i][energydensities_dm<max_rhoc_dm] = dm_pres_rho
                dm_pres_rho_interp = UnivariateSpline(energydensities_dm[energydensities_dm<max_rhoc_dm], dm_pres_rho, k=1, s=0, ext = 1)

                #Only want see impact of ADM where we have baryonic central desnities
                pressures_rho[:,i][energydensities_b<max_rhoc] = pressures_rho_b[:,i][energydensities_b<max_rhoc] + dm_pres_rho_interp(energydensities_b[energydensities_b<max_rhoc])

            else:
                pressures_rho[:,i][energydensities_b<max_rhoc] = pressures_rho_b[:,i][energydensities_b<max_rhoc]



            pressures_b[:,i][energydensities_b<EOS.max_edsc] = EOS.eos(energydensities_b[energydensities_b<EOS.max_edsc])
            dm_pres_eps = eos_dm(energydensities_dm[energydensities_dm<max_edsc_dm])

            if len(dm_pres_eps) != 0:
                pressures_dm[:,i][energydensities_dm<max_edsc_dm] = dm_pres_eps
                dm_pres_eps_interp = UnivariateSpline(energydensities_dm[energydensities_dm<max_edsc_dm], dm_pres_eps, k=1, s=0, ext = 1)

                #Only want to see impact of ADM where we have baryonic central energy densities
                pressures[:,i][energydensities_b<EOS.max_edsc] = pressures_b[:,i][energydensities_b<EOS.max_edsc] + dm_pres_eps_interp(energydensities_b[energydensities_b<EOS.max_edsc])

            else:
                pressures[:,i][energydensities_b<EOS.max_edsc] = pressures_b[:,i][energydensities_b<EOS.max_edsc]

            for j, e in enumerate(rhocs):
                epsdm = EOS.find_epsdm_cent(EOS.adm_fraction,e)
                rhocsdm[j] = epsdm
                star = Star(e,epsdm)
                star.solve_structure(EOS.energydensities, EOS.pressures,EOS.energydensities_dm, EOS.pressures_dm, EOS.dm_halo)
                M[j] = star.Mrot
                R[j] = star.Req

                if EOS.reach_fraction == False:
                    R[j] = 0.0
                    rhocsdm[j] = 0.0



            idx_no_zero = np.nonzero(R)
            M = M[idx_no_zero]
            R = R[idx_no_zero]
            rhocs = rhocs[idx_no_zero]
            rhocsdm = rhocsdm[idx_no_zero]

            M, indices = np.unique(M, return_index=True)
            try:
                MR = UnivariateSpline(M, R[indices], k=1, s=0, ext=1)

                rhocs_admixed = rhocs[indices] + rhocsdm[indices]
                rhocM = UnivariateSpline(M, rhocs_admixed, k=1, s=0, ext = 1)
            except Exception:
                MR = 0
                rhocs_admixed = rhocs[indices] + rhocsdm[indices]
                rhocM = 0

            rhoc = np.random.rand() *(np.log10(EOS.max_edsc) - 14.6) + 14.6
            epsdm = EOS.find_epsdm_cent(EOS.adm_fraction,10**rhoc)
            star = Star(10**rhoc,epsdm)
            star.solve_structure(EOS.energydensities, EOS.pressures,EOS.energydensities_dm, EOS.pressures_dm, EOS.dm_halo)
            mass_radius[i] = star.Mrot, star.Req

            if EOS.reach_fraction == False:
                mass_radius[i] = star.Mrot, 0.0

            for j, e in enumerate(rhocpar):
                epsdm = EOS.find_epsdm_cent(EOS.adm_fraction,e)
                star = Star(e,epsdm)  #two_fluid_tidal not used here since chirp_masses are none, so compute two fluid tidal is waste of time
                star.solve_structure(EOS.energydensities, EOS.pressures,EOS.energydensities_dm, EOS.pressures_dm, EOS.dm_halo)
                if EOS.reach_fraction == True:
                    tmp.append([e + epsdm, EOS.eos(e) + eos_dm(epsdm), star.Mrot, star.Req, star.Rdm_halo])

                if chirp_masses[j] is not None:
                    if rhocM != 0:
                        M2 = m1(chirp_masses[j], tmp[j][4])
                        rhoc = rhocM(M2)
                        epsdm = EOS.find_epsdm_cent(EOS.adm_fraction,rhoc)
                        star = Star(rhoc,epsdm)  #two_fluid_tidal used here since we need star.tidal
                        star.solve_structure(EOS.energydensities, EOS.pressures,EOS.energydensities_dm, EOS.pressures_dm, EOS.dm_halo, EOS.two_fluid_tidal)

                        if EOS.reach_fraction == True:
                            tmp.append([e + epsdm, EOS.eos(e) + eos_dm(epsdm), star.Mrot, star.Req, star.Rdm_halo])

            if tmp != []:
                scattered.append(tmp)

            if MR != 0:
                radii[:,i] = MR(masses)

    mass_radius = mass_radius[mass_radius[:,1] != 0]

    # Save everything
    scattered = np.array(scattered)
    savedata = [pressures, radii, scattered, mass_radius]
    fnames = ['pressures.npy', 'radii.npy', 'scattered.npy', 'MR_prpr.txt'] # TODO change the name of MR_prpr.txt? Check with the team

    if dm:
        fnames += ['pressures_baryon.npy', 'pressures_dm.npy']
        savedata += [pressures_b, pressures_dm]
    if not flag:
        minradii, maxradii = calc_bands(masses, radii)
        savedata += [minradii, maxradii]
        fnames += ['minradii.npy', 'maxradii.npy']
        if dm:
            minpres, maxpres = calc_bands(energydensities_b, pressures)
            minpres_rho, maxpres_rho = calc_bands(energydensities_b, pressures_rho)
            minpres_b, maxpres_b = calc_bands(energydensities_b, pressures_b)
            minpres_dm, maxpres_dm = calc_bands(energydensities_dm, pressures_dm)
            savedata += [minpres_rho, maxpres_rho, minpres, maxpres, minpres_b, maxpres_b, minpres_dm, maxpres_dm]
            fnames += ['minpres_rho.npy', 'maxpres_rho.npy', 'minpres.npy', 'maxpres.npy']
            fnames += ['minpres_baryon.npy', 'maxpres_baryon.npy', 'minpres_dm.npy', 'maxpres_dm.npy']
        else:
            minpres, maxpres = calc_bands(energydensities, pressures)
            minpres_rho, maxpres_rho = calc_bands(energydensities, pressures_rho)
            savedata += [minpres_rho, maxpres_rho, minpres, maxpres]
            fnames += ['minpres_rho.npy', 'maxpres_rho.npy', 'minpres.npy', 'maxpres.npy']
    save_auxiliary_data(path, identifier, savedata, fnames)

def cornerplot(root_name, variable_params, dm = False): #Add ADM functionality
    ewposterior = np.loadtxt(root_name + 'post_equal_weights.dat')
    if dm == False:
        figure = corner.corner(ewposterior[:,0:-1], labels = list(variable_params.keys()), show_titles=True,
                        color=colors[4], quantiles =[0.16, 0.5, 0.84], smooth=.8)
    else:
        idx_mchi = list(variable_params.keys()).index('mchi')
        idx_gchi_over_mphi = list(variable_params.keys()).index('gchi_over_mphi')

        ewposterior[:,idx_mchi] = np.log10(ewposterior[:,idx_mchi])
        ewposterior[:,idx_gchi_over_mphi] = np.log10(ewposterior[:,idx_gchi_over_mphi])

        figure = corner.corner(ewposterior[:,0:-1], labels = list(variable_params.keys()), show_titles=True,
                        color=colors[4], quantiles =[0.16, 0.5, 0.84], smooth=.8)

    figure.savefig(root_name + 'corner.png')

def mass_radius_posterior_plot(root_name):
    scatter = np.load(root_name + 'scattered.npy')
    figure, ax = plt.subplots(1,1, figsize=(9,6))
    M_max = 0.
    for i in range(len(scatter[0])):
        corner.hist2d(scatter[:,i][:,3], scatter[:,i][:,2], labels = ['R [km]', r'M [M$_{\odot}$]'], show_titles=True,
                        color=colors[i], smooth=.8, data_kwargs={'ms':5, 'alpha':0.5})
        M_max = max(max(scatter[:,i][:,2]), M_max)

    ax.set_xlim(8, 15)
    ax.set_ylim(1., M_max)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'Radius [km]', fontsize=15)
    ax.set_ylabel(r'Mass [M$_{\odot}$]', fontsize=15)
    plt.tight_layout()
    plt.show()
    figure.savefig(root_name + 'MRposterior.png')

def mass_radius_prior_predictive_plot(root_name,variable_params, label_name='updated prior'):
    fig, ax = plt.subplots(1,1, figsize=(9, 6))

    num_stars = len(np.array([v for k,v in variable_params.items() if 'rhoc' in k]))

    if len(list(variable_params.keys())) == num_stars:
        flag = True

    else:
        flag = False

    if flag == True:
        raise Exception("Cannot perform mass_radius_prior_predictive_plot function because EoS is fixed, i.e., tabulated or all EoS params are static params!")
    else:
        MR_prpr= np.loadtxt(root_name + 'MR_prpr.txt')

        inbins = np.histogramdd(MR_prpr[:,[1,0]], bins=50, density=True)
        levels = [0.05, 0.32, 1]

        sns.kdeplot(x=MR_prpr[:,1], y=MR_prpr[:,0], gridsize=50, fill=True, ax=ax, levels=levels,
                    alpha=1., cmap=ListedColormap(colors[[5,3]]))

        # add legend, for now 'prior' is always shown
        custom_lines = [Line2D([0], [0], color='black', lw=2, linestyle='--', alpha=1.),
                    mpatches.Patch(color=colors[5], alpha=1.)]
        ax.legend(custom_lines, ['Prior', label_name],
                    loc=1, prop={'size': 14})

        ax.yaxis.set_ticks([1., 1.5, 2., 2.5, 3.0])
        ax.set_ylabel(r"M (M$_{\odot}$)", fontsize=20)
        ax.tick_params(top=1,right=1, which='both', direction='in', labelsize=20)
        ax.set_xlabel(r"R (km)", fontsize=20)
        ax.set_ylim(1., 3.)
        ax.set_xlim(9.05, 15)
        plt.tight_layout()
        fig.savefig(root_name + 'MRpriorpredictive.png')

def eos_posterior_plot(root_name,variable_params, prior_contours=None, dm = False):
    """
    Function to plot the p-eps posteriors.


    Parameters
    ----------

    root_name: str
        Name of the inference run to refer back to. Used to get the Multinest outputs.

    variable_params: dict
        Variable parameters in the inference script.

    prior_contours: bool
        If True, include the prior contours

    dm: bool
        If True, ADM is included in the p-eps posteriors. Otherwise, the baryonic only p-eps is used.


    """
    fig, ax = plt.subplots(1,1, figsize=(9, 6))
    my_fontsize=20

    num_stars = len(np.array([v for k,v in variable_params.items() if 'rhoc' in k]))

    if len(list(variable_params.keys())) == num_stars:
        flag = True

    else:
        flag = False

    if flag == True:
        raise Exception("Cannot perform mass_radius_prior_predictive_plot function because EoS is fixed, i.e., tabulated or all EoS params are static params!")
    else:
        if dm == True:
            minpres_pp = np.log10(np.load(root_name + 'minpres.npy'))
            maxpres_pp = np.log10(np.load(root_name + 'maxpres.npy'))

        else:
            minpres_pp = np.log10(np.load(root_name + 'minpres_baryon.npy'))
            maxpres_pp = np.log10(np.load(root_name + 'maxpres_baryon.npy'))

        scatter = np.load(root_name + 'scattered.npy')
        central_density_post = np.log10(scatter[:,0][:,[0,1]])

        corner.hist2d(central_density_post[:,0], central_density_post[:,1], show_titles=False,
                            color=colors[3], plot_data_points=False, plot_density=False,
                    levels=[0.68, 0.95])


        ax.fill_between(minpres_pp[:,0], minpres_pp[:,2], maxpres_pp[:,2],
                            color=sns.cubehelix_palette(8, start=.5, rot=-.75, dark=.2, light=.85)[0], alpha=1)
        ax.fill_between(minpres_pp[:,0], minpres_pp[:,1], maxpres_pp[:,1],
                            color=sns.cubehelix_palette(8, start=.5, rot=-.75, dark=.2, light=.85)[3], alpha=1)
        if prior_contours is not None:
            minpres_prior = np.log10(np.load(prior_contours))
            maxpres_prior = np.log10(np.load(prior_contours))

            ax.plot(maxpres_prior[:,0], minpres_prior[:,2], c='black', linestyle='--', lw=2)
            ax.plot(maxpres_prior[:,0], maxpres_prior[:,2], c='black', linestyle='--', lw=2)

        ax.set_ylabel(r'$\log_{10}(P)$ (dyn/cm$^2$)', fontsize=my_fontsize)
        ax.set_xlabel(r'$\log_{10}(\varepsilon)$ (g/cm$^3$)', fontsize=my_fontsize)
        ax.tick_params(top=1,right=1, which='both', direction='in', labelsize=my_fontsize)

        ax.set_xlim(14.25, max(minpres_pp[:,0]))
        ax.set_ylim(33, 36.2)

        plt.tight_layout()
        fig.savefig(root_name + 'EoSposterior.png')
