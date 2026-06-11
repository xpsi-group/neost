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
from tqdm import tqdm
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

    equal_weighted_samples = np.loadtxt(sample_file, skiprows=skiprows, dtype=np.float128)
    equal_weighted_sampled = np.float64(equal_weighted_samples)

    return equal_weighted_sampled

def save_auxiliary_data(path, identifier, data):
    for fname, value in data.items():
        extension = pathlib.Path(fname).suffix.lower()
        savefunc = np.savetxt if extension == '.txt' else np.save
        fname = f'{path}/{identifier}{fname}'
        savefunc(fname, value)
        print(f'Writing {fname} to disk')

def print_samples_per_core(samples):
    print('Number of samples to be analyzed per core:', end=' ')
    for a in samples:
        print(len(a), end=' ')
    print()

def recast_equal_weighted_samples_for_mpi(equal_weighted_samples, num_processes):
    num_samples = len(equal_weighted_samples)
    samples = [[] for i in range(num_processes)]
    samples_per_core = int(np.ceil(num_samples / num_processes))
    for current_core in range(num_processes):
        for i in range(samples_per_core):
            idx = current_core*samples_per_core + i
            if idx >= num_samples:
                break
            samples[current_core].append(equal_weighted_samples[idx])
    return samples


def compute_auxiliary_data_Thijs(path, EOS, variable_params, static_params, chirp_masses, prior = False, sampler='multinest', identifier=''):
    """
    Function to compute the posterior auxiliary data used to generate standard NEoST plots, such as, the pressures, (if dm = True)
    the baryonic pressure, mass-radius posteriors, and p-eps posteriors.


    Parameters
    ----------

    path: str
        The path to where the sampler output is stored.

    EOS: obj
        equation of state object initialized in the inference script, i.e., the parameters that are sampled during inferencing.

    variable_params: dict
        Variable parameters in the inference script.

    static_params: dict
        Static parameters in the inference script, i.e., the parameters that are held static during inference sampling.

    chirp_masses: list
        List determining if GW data is included. If None, just MR only.


    sampler: str
        The sampler used, either 'multinest' or 'ultranest'.

    identifier: str
        The name given to the sampling. So the sampler output would be called something like <path>/<identifier>post_equal_weights.dat.
        Only used with Multinest, ignored for Ultranest.

    """
    # Set up some MPI things
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank() # The rank of the current MPI process
    num_processes = comm.Get_size() # Number of MPI processes
    samples = None # This is essentially a rearranged 'equal_weighted_samples'

    # Get number of stars and set 'eos_is_fixed' that avoids unneeded calculations
    num_stars = len(np.array([v for k,v in variable_params.items() if 'rhoc' in k]))
    eos_is_fixed = True if len(list(variable_params.keys())) == num_stars else False

    if mpi_rank == 0:
        equal_weighted_samples = load_equal_weighted_samples(path, sampler, identifier)
        num_samples = len(equal_weighted_samples)
        num_stars = len(np.array([v for k,v in variable_params.items() if 'rhoc' in k]))
        print(f"Total number of samples is {num_samples}, and the number of stars is {num_stars}")

        # Recast equal_weighted_samples in a form suitable for MPI scatter
        samples = recast_equal_weighted_samples_for_mpi(equal_weighted_samples, num_processes)
        print_samples_per_core(samples)

    # Scatter samples to the different processes and compute
    samples = comm.scatter(samples, root=0)
    results = _compute_auxiliary_data_thread_Thijs(samples, EOS, variable_params, static_params, chirp_masses, prior, eos_is_fixed, mpi_rank)

    # Gather
    results = comm.gather(results, root=0)

    if mpi_rank == 0:
        masses = results[0].get('masses')
        energydensities = results[0].get('energydensities')
        mass_radius = np.concatenate([result.get('mass_radius') for result in results])
        mass_moi = np.concatenate([result.get('mass_moi') for result in results])
        radii = np.concatenate([result.get('radii') for result in results], axis=1)
        pressures = np.concatenate([result.get('pressures') for result in results], axis=1)
        pressures_rho = np.concatenate([result.get('pressures_rho') for result in results], axis=1)
        scattered = np.concatenate([result.get('scattered') for result in results])
        
        # Filter out unphysical results
        mass_radius = mass_radius[mass_radius[:,1] != 0]
        mass_moi = mass_moi[mass_moi[:,1] != 0]

        # Save everything
        savedata = {'pressures.npy':pressures, 'energydensities.npy':energydensities, 'radii.npy':radii, 'scattered.npy':scattered, 'MR_prpr.txt':mass_radius, 'mass_moi.npy':mass_moi}

        if not eos_is_fixed:
            minradii, maxradii = calc_bands(masses, radii)
            savedata['minradii.npy'] = minradii
            savedata['maxradii.npy'] = maxradii

            minpres, maxpres = calc_bands(energydensities, pressures)
            minpres_rho, maxpres_rho = calc_bands(energydensities, pressures_rho)
            savedata['minpres_rho.npy'] = minpres_rho
            savedata['maxpres_rho.npy'] = maxpres_rho
            savedata['minpres.npy'] = minpres
            savedata['maxpres.npy'] = maxpres
        save_auxiliary_data(path, identifier, savedata)

def _compute_auxiliary_data_thread_Thijs(samples, EOS, variable_params, static_params, chirp_masses, prior, eos_is_fixed, thread_number):
    '''
    Here the calculations of auxiliary data is done.
    Reading/writing of files and parallelization is done by compute_auxiliary_data_Thijs(),
    this function just calculates and returns. Not meant to be called manually.
    '''
    num_samples = len(samples)
    print(f'MPI-process {thread_number} is computing auxiliary data for {num_samples} samples ...')

    # Grids
    # More points are added to account for larger energy density spread from ADM
    # total ADM [1e12,1e18] + baryonic energy densities [1e14.2,1e16]
    num_grid_points = 50
    masses = np.linspace(.2, 2.9, num_grid_points)
    energydensities = np.logspace(14.2, 16, num_grid_points)

    mass_radius = np.zeros((num_samples, 2))
    mass_moi = np.zeros((num_samples, 2))
    radii = np.zeros((num_grid_points, num_samples))
    pressures = np.zeros((num_grid_points, num_samples))
    pressures_rho = np.zeros((num_grid_points, num_samples))
    scattered = []


    for i in tqdm(range(0, num_samples)):
        pr = samples[i][0:len(variable_params)]
        par = {e:pr[j] for j, e in enumerate(list(variable_params.keys()))}
        par.update(static_params)
        EOS.update(par, max_edsc=True)

        rhocs = np.logspace(14.5, np.log10(EOS.max_edsc), 30)

        M = np.zeros(len(rhocs))
        R = np.zeros(len(rhocs))

        rhocpar = np.array([10**v for k,v in par.items() if 'rhoc' in k])
        scattered_elements = []

        rhopres = UnivariateSpline(EOS.massdensities, EOS.pressures, k=1, s=0)
        edsrho = UnivariateSpline(EOS.energydensities, EOS.massdensities, k=1, s=0)
        max_rhoc = edsrho(EOS.max_edsc)
        indices = energydensities<max_rhoc
        pressures_rho[:,i][indices] = rhopres(energydensities[indices])
        indices = energydensities<EOS.max_edsc
        pressures[:,i][indices] = EOS.eos(energydensities[indices])

        for j, e in enumerate(rhocs):
            star = Star(e)
            star.solve_structure(EOS.energydensities, EOS.pressures)
            M[j] = star.Mrot
            R[j] = star.Req

        indices = np.diff(M) > 0
        indices = np.insert(indices, 0, True)
        MR = UnivariateSpline(M[indices], R[indices], k=1, s=0, ext=1)
        rhocM = UnivariateSpline(M[indices], rhocs[indices], k=1, s=0, ext=1)

        for j, e in enumerate(rhocpar):
            star = Star(e)
            star.solve_structure(EOS.energydensities, EOS.pressures)
            scattered_elements.append([e, EOS.eos(e), star.Mrot, star.Req, star.tidal, star.moi45])

            if chirp_masses[j] is not None:
                M2 = m1(chirp_masses[j], scattered_elements[j][2])
                rhoc = rhocM(M2)
                star = Star(rhoc)
                star.solve_structure(EOS.energydensities, EOS.pressures)
                scattered_elements.append([rhoc, EOS.eos(rhoc), star.Mrot, star.Req, star.tidal, star.moi45])

        scattered.append(scattered_elements)
        radii[:,i] = MR(masses)
        if prior == False:
            rhoc = np.random.rand() *(np.log10(EOS.max_edsc) - 14.6) + 14.6
            star = Star(10**rhoc)
            star.solve_structure(EOS.energydensities, EOS.pressures)
            mass_radius[i] = star.Mrot, star.Req
            mass_moi[i] = star.Mrot, star.moi45
        else:
            rhoc = par['rhoc_1']
            star = Star(10**rhoc)
            star.solve_structure(EOS.energydensities, EOS.pressures)
            mass_radius[i] = star.Mrot, star.Req
            mass_moi[i] = star.Mrot, star.moi45
        
    return_values = {'pressures':pressures, 'pressures_rho':pressures_rho, 'masses':masses, 'radii':radii, 'scattered':scattered, 'mass_radius':mass_radius, 'energydensities':energydensities, 'mass_moi':mass_moi}
    return return_values



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
        if dm == False:
            minpres_pp = np.log10(np.load(root_name + 'minpres.npy'))
            maxpres_pp = np.log10(np.load(root_name + 'maxpres.npy'))

        else:
            minpres_pp = np.log10(np.load(root_name + 'minpres_baryon.npy'))
            maxpres_pp = np.log10(np.load(root_name + 'maxpres_baryon.npy'))

        scatter = np.load(root_name + 'scattered.npy')
        central_density_post = np.log10(scatter[:,3][:,[0,1]])

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
