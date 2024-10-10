# 3rd party
import numpy as np
import kalepy
from scipy.stats import gaussian_kde

def get_likelihood(case, data_path):
    if 'prior' == case:
        return get_likelihood_prior() # Returns an empty likelihood
    elif 'baseline' == case:
        return get_likelihood_baseline(data_path)
    elif 'new' == case:
        return get_likelihood_new(data_path)
    elif 'new2' == case:
        return get_likelihood_new2(data_path)
    elif 'new3' == case:
        return get_likelihood_new3(data_path)
    else:
        raise ValueError(f"Only 'baseline', 'new', 'new2' and 'new3' currently implemented. Unknown case {case}")

def get_likelihood_prior():
    likelihood_functions = []
    likelihood_params = [['Mass', 'Radius']]
    chirp_mass = [None]
    return likelihood_functions, likelihood_params, chirp_mass

def get_likelihood_baseline(data_path):
    # Create the likelihoods for the individual measurements

    mr_J0740 = np.loadtxt(f'{data_path}/J0740_Salmi22_3C50-3X_STU_lp40k_se01_mrsamples_post_equal_weights.dat').T
    J0740_LL = gaussian_kde(mr_J0740)

    mr_J0030 = np.loadtxt(f'{data_path}/J0030_bravo_STPST_lp10k_se03_MM_mrsamples_post_equal_weights.dat').T
    J0030_LL = gaussian_kde(mr_J0030)

    # And next up is the gravitational wave event
    GW170817 = np.load(f'{data_path}/GW170817_McQL1L2weights.npy')
    GW170817_LL = kalepy.KDE(GW170817[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW170817[:,4], bandwidth=0.1, kernel='gaussian')

    GW190425 = np.load(f'{data_path}/GW190425_McQL1L2weights.npy')
    GW190425_LL = kalepy.KDE(GW190425[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW190425[:,4], bandwidth=0.1, kernel='gaussian')

    # Pass the likelihoods to the solver
    likelihood_functions = [J0740_LL.pdf, J0030_LL.pdf, lambda points: GW170817_LL.density(np.array([points]).T, probability=True)[1][0], lambda points: GW190425_LL.density(np.array([points]).T, probability=True)[1][0]]
    likelihood_params = [['Mass', 'Radius'], ['Mass', 'Radius']]

    # Define whether event is GW or not and define number of stars/events
    chirp_mass = [None,None,1.186,1.44]

    # Return values
    return likelihood_functions, likelihood_params, chirp_mass


def get_likelihood_new(data_path):
    # Create the likelihoods for the individual measurements
    mr_J0740 = np.loadtxt(f'{data_path}/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat').T
    J0740_LL = gaussian_kde(mr_J0740)

    mr_J0030 = np.loadtxt(f'{data_path}/J0030_bravo_STPDT_NxX_lp1k_se08_mrsamples_post_equal_weights.dat').T
    J0030_LL = gaussian_kde(mr_J0030)

    mr_J0437 = np.loadtxt(f'{data_path}/J0437_3C50_CST_PDT_AGN_lp20k_se03_mrsamples_post_equal_weights.dat').T
    J0437_LL = gaussian_kde(mr_J0437)

    # And next up is the gravitational wave event
    GW170817 = np.load(f'{data_path}/GW170817_McQL1L2weights.npy')
    GW170817_LL = kalepy.KDE(GW170817[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW170817[:,4], bandwidth=0.1, kernel='gaussian')

    GW190425 = np.load(f'{data_path}/GW190425_McQL1L2weights.npy')
    GW190425_LL = kalepy.KDE(GW190425[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW190425[:,4], bandwidth=0.1, kernel='gaussian')

    # Pass the likelihoods to the solver
    likelihood_functions = [J0740_LL.pdf, J0030_LL.pdf, J0437_LL.pdf, lambda points: GW170817_LL.density(np.array([points]).T, probability=True)[1][0],lambda points: GW190425_LL.density(np.array([points]).T, probability=True)[1][0]]
    likelihood_params = [['Mass', 'Radius'], ['Mass', 'Radius'], ['Mass', 'Radius']]

    # Define whether event is GW or not and define number of stars/events
    chirp_mass = [None,None,None,1.186,1.44]

    # Return values
    return likelihood_functions, likelihood_params, chirp_mass


def get_likelihood_new2(data_path):
    # Create the likelihoods for the individual measurements
    mr_J0740 = np.loadtxt(f'{data_path}/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat').T
    J0740_LL = gaussian_kde(mr_J0740)

    mr_J0030 = np.loadtxt(f'{data_path}/J0030_bravo_STPST_lp10k_se03_MM_mrsamples_post_equal_weights.dat').T
    J0030_LL = gaussian_kde(mr_J0030)

    mr_J0437 = np.loadtxt(f'{data_path}/J0437_3C50_CST_PDT_AGN_lp20k_se03_mrsamples_post_equal_weights.dat').T
    J0437_LL = gaussian_kde(mr_J0437)

    # And next up is the gravitational wave event
    GW170817 = np.load(f'{data_path}/GW170817_McQL1L2weights.npy')
    GW170817_LL = kalepy.KDE(GW170817[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW170817[:,4], bandwidth=0.1, kernel='gaussian')

    GW190425 = np.load(f'{data_path}/GW190425_McQL1L2weights.npy')
    GW190425_LL = kalepy.KDE(GW190425[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW190425[:,4], bandwidth=0.1, kernel='gaussian')

    # Pass the likelihoods to the solver
    likelihood_functions = [J0740_LL.pdf, J0030_LL.pdf, J0437_LL.pdf, lambda points: GW170817_LL.density(np.array([points]).T, probability=True)[1][0],lambda points: GW190425_LL.density(np.array([points]).T, probability=True)[1][0]]
    likelihood_params = [['Mass', 'Radius'], ['Mass', 'Radius'], ['Mass', 'Radius']]

    # Define whether event is GW or not and define number of stars/events
    chirp_mass = [None,None,None,1.186,1.44]

    # Return values
    return likelihood_functions, likelihood_params, chirp_mass


def get_likelihood_new3(data_path):
    # Create the likelihoods for the individual measurements
    mr_J0740 = np.loadtxt(f'{data_path}/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat').T
    J0740_LL = gaussian_kde(mr_J0740)

    mr_J0030 = np.loadtxt(f'{data_path}/J0030_bravo_PDTU_NxX_lp1k_se08_mrsamples_post_equal_weights.dat').T
    J0030_LL = gaussian_kde(mr_J0030)

    mr_J0437 = np.loadtxt(f'{data_path}/J0437_3C50_CST_PDT_AGN_lp20k_se03_mrsamples_post_equal_weights.dat').T
    J0437_LL = gaussian_kde(mr_J0437)

    # And next up is the gravitational wave event
    GW170817 = np.load(f'{data_path}/GW170817_McQL1L2weights.npy')
    GW170817_LL = kalepy.KDE(GW170817[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW170817[:,4], bandwidth=0.1, kernel='gaussian')

    GW190425 = np.load(f'{data_path}/GW190425_McQL1L2weights.npy')
    GW190425_LL = kalepy.KDE(GW190425[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW190425[:,4], bandwidth=0.1, kernel='gaussian')

    # Pass the likelihoods to the solver
    likelihood_functions = [J0740_LL.pdf, J0030_LL.pdf, J0437_LL.pdf, lambda points: GW170817_LL.density(np.array([points]).T, probability=True)[1][0],lambda points: GW190425_LL.density(np.array([points]).T, probability=True)[1][0]]
    likelihood_params = [['Mass', 'Radius'], ['Mass', 'Radius'], ['Mass', 'Radius']]

    # Define whether event is GW or not and define number of stars/events
    chirp_mass = [None,None,None,1.186,1.44]

    # Return values
    return likelihood_functions, likelihood_params, chirp_mass
