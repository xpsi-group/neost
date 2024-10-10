
Basically here is the data used for the posterior runs. 

The easiest thing to get out of the way is the neutron star measurements.

1. Each neutron star maeasurement has two columns, the first being mass posterior samples and the second being radius posterior samples.
2. To figure out which pulsar measurment to use see the file name a compare with Table 1 in the overleaf. For example, J0030 NICER ST+PST in Table 1 corresponds to J0030_bravo_STPST_lp10k_se03_MM_mrsamples_post_equal_weights.dat


To use them in your posterior script copy and paste the folling:
1. # Create the likelihoods for the individual measurements
mr_J0740 = np.loadtxt('J0740_Salmi22_3C50-3X_STU_lp40k_se01_mrsamples_post_equal_weights.dat').T
J0740_LL = gaussian_kde(mr_J0740)

# Pass the likelihoods to the solver
likelihood_functions = [J0740_LL.pdf]


For the GW files, here is what you need to know:

1. first column: chirp mass in the source frame
2. second column: mass-ratio (m2/m1)
3. third column: tidal def of mass 1
4. fourth column: tidal def of mass 2
5. fifth column: weights of each sample to ensure that the Ligo/Virgo prior on chirp mass and mass-ratio is uniform.

To use the GW files in the posterior script, copy and paste the following:
1. # And next up is the gravitational wave event
GW170817 = np.load('GW170817_McQL1L2weights.npy')
GW170817_LL = kalepy.KDE(GW170817[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW170817[:,4], bandwidth=0.1, kernel='gaussian')

#reflect is simply a way to reflect the boundaries in the event so for example your kde stays uniform.
#see https://kalepy.readthedocs.io/en/latest/kde_api.html#resampling-constructing-statistically-similar-values for further detials under Fancy Usage


# Pass the likelihoods to the solver
likelihood_functions = [lambda points: GW170817_LL.density(np.array([points]).T, probability=True)[1][0]]


To use mass-radius and GWs together in the posterior script see the following example below for the baseline posterior case:

# Create the likelihoods for the individual measurements
mr_J0740 = np.loadtxt('J0740_Salmi22_3C50-3X_STU_lp40k_se01_mrsamples_post_equal_weights.dat').T
J0740_LL = gaussian_kde(mr_J0740)

mr_J0030 = np.loadtxt('J0030_bravo_STPST_lp10k_se03_MM_mrsamples_post_equal_weights.dat').T
J0030_LL = gaussian_kde(mr_J0030)


# And next up is the gravitational wave event
GW170817 = np.load('GW170817_McQL1L2weights.npy')
GW170817_LL = kalepy.KDE(GW170817[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW170817[:,4], bandwidth=0.1, kernel='gaussian')

GW190425 = np.load('GW190425_McQL1L2weights.npy')
GW190425_LL = kalepy.KDE(GW190425[:,0:4].T, reflect=[[None, None], [None, 1.], [0., None], [0., None]], weights=GW190425[:,4], bandwidth=0.1, kernel='gaussian')


# Pass the likelihoods to the solver
likelihood_functions = [J0740_LL.pdf, J0030_LL.pdf, lambda points: GW170817_LL.density(np.array([points]).T, probability=True)[1][0],
                        lambda points: GW190425_LL.density(np.array([points]).T, probability=True)[1][0]]
likelihood_params = [['Mass', 'Radius'], ['Mass', 'Radius']]

# Define whether event is GW or not and define number of stars/events
chirp_mass = [None,None,1.186,1.44]
number_stars = len(chirp_mass)

