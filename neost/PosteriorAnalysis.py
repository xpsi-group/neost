import numpy
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import pyplot
import seaborn as sns
import getdist
from getdist import plots
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
from alive_progress import alive_bar
import corner

import neost
from neost.eos import polytropes, tabulated
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
import neost.global_imports as global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons

colors = numpy.array(["#c6878f", "#b79d94", "#969696", "#67697c", "#233b57", "#BCBEC7"])

dyncm2_to_MeVfm3 = 1./(1.6022e33)
gcm3_to_MeVfm3 = 1./(1.7827e12)
oneoverfm_MeV = 197.33


def m1(mc, m2):
    num1 = (2./3.)**(1./3.)*mc**5.
    denom1 = (9*m2**7. *mc**5. + numpy.sqrt(3.)*numpy.sqrt(abs(27*m2**14.*mc**10.-4.*m2**9.*mc**15.)))**(1./3.)
    denom2 = 2.**(1./3.)*3.**(2./3.)*m2**3.
    return num1/denom1 + denom1/denom2

def calc_bands(x, y):
    miny = numpy.zeros((len(y),3))
    maxy = numpy.zeros((len(y),3))
    
    for i in range(len(y)):
        z = y[i][y[i]>0.0]
        if len(z)<200:
            print('sample too small for %.2f' %x[i])
            continue
        kde = gaussian_kde(z)
        testz = numpy.linspace(min(z),max(z), 1000)
        pdf = kde.pdf(testz)
        array = pdf
        index_68 = numpy.where(numpy.cumsum(numpy.sort(array)[::-1]) < sum(array)*0.6827)[0]
        index_68 = numpy.argsort(array)[::-1][index_68]
        index_95 = numpy.where(numpy.cumsum(numpy.sort(array)[::-1]) < sum(array)*0.95)[0]
        index_95 = numpy.argsort(array)[::-1][index_95]
        miny[i] =  x[i], min(testz[index_68]), min(testz[index_95])
        maxy[i] =  x[i], max(testz[index_68]), max(testz[index_95])
        
    miny = miny[~numpy.all(miny == 0, axis=1)]
    maxy = maxy[~numpy.all(maxy == 0, axis=1)]
    return miny, maxy


def compute_auxiliary_data(root_name, EOS, variable_params, static_params, chirp_masses): 
    ewposterior = numpy.loadtxt(root_name + 'post_equal_weights.dat')
    print("Total number of samples is %d" %(len(ewposterior)))

    num_stars = len(numpy.array([v for k,v in variable_params.items() if 'rhoc' in k]))

    if len(list(variable_params.keys())) == num_stars:
        flag = True

    else: 
        flag = False

    masses = numpy.linspace(.2, 2.9, 50)
    energydensities = numpy.logspace(14.2, 16, 50)
    scattered = []

    if flag == True:
        radii = numpy.zeros((len(masses), len(ewposterior)))
        pressures = numpy.zeros((len(masses), len(ewposterior)))
        pressures_rho = numpy.zeros((len(masses), len(ewposterior)))
        MR_prpr_pp = numpy.zeros((len(ewposterior), 2))

    else:
        radii = numpy.zeros((len(masses), len(ewposterior)))
        pressures = numpy.zeros((len(masses), len(ewposterior)))
        pressures_rho = numpy.zeros((len(masses), len(ewposterior)))
        minradii = numpy.zeros((3, len(masses)))
        maxradii = numpy.zeros((3, len(masses)))
        minpres = numpy.zeros((3, len(energydensities)))
        maxpres = numpy.zeros((3, len(energydensities)))
        minpres_rho = numpy.zeros((3, len(energydensities)))
        maxpres_rho = numpy.zeros((3, len(energydensities)))
        MR_prpr_pp = numpy.zeros((len(ewposterior), 2))

    
    


    with alive_bar(len(ewposterior)) as bar:
        for i in range(0, len(ewposterior), 1):

            pr = ewposterior[i][0:len(variable_params)]
            par = {e:pr[j] for j, e in enumerate(list(variable_params.keys()))}
            par.update(static_params)
            EOS.update(par, max_edsc=True)

            rhopres = UnivariateSpline(EOS.massdensities, EOS.pressures, k=1, s=0)
            edsrho = UnivariateSpline(EOS.energydensities, EOS.massdensities, k=1, s=0)
            max_rhoc = edsrho(EOS.max_edsc)
            pressures_rho[:,i][energydensities<max_rhoc] = rhopres(energydensities[energydensities<max_rhoc]
            pressures[:,i][energydensities<EOS.max_edsc] = EOS.eos(energydensities[energydensities<EOS.max_edsc])
            
            rhocs = numpy.logspace(14.5, numpy.log10(EOS.max_edsc), 30)
            M = numpy.zeros(len(rhocs))
            R = numpy.zeros(len(rhocs))
            for j, e in enumerate(rhocs):
                star = Star(e)
                star.solve_structure(EOS.energydensities, EOS.pressures)
                M[j] = star.Mrot
                R[j] = star.Req

            M, indices = numpy.unique(M, return_index=True)
            MR = UnivariateSpline(M, R[indices], k=1, s=0, ext=1)
            rhocM = UnivariateSpline(M, rhocs[indices], k=1, s=0)
            
            rhocpar = numpy.array([10**v for k,v in par.items() if 'rhoc' in k])
            tmp = []
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
            rhoc = numpy.random.rand() *(numpy.log10(EOS.max_edsc) - 14.6) + 14.6
            star = Star(10**rhoc)
            star.solve_structure(EOS.energydensities, EOS.pressures)
            MR_prpr_pp[i] = star.Mrot, star.Req

            radii[:,i] = MR(masses)
            bar()

    scattered = numpy.array(scattered)
    # save everything
    numpy.save(root_name + 'pressures', pressures)
    numpy.save(root_name + 'radii', radii)
    numpy.save(root_name + 'scattered', scattered)
    numpy.savetxt(root_name + 'MR_prpr.txt', MR_prpr_pp)
   
    if flag == False:
        minpres, maxpres = calc_bands(energydensities, pressures)
        minpres_rho, maxpres_rho = calc_bands(energydensities, pressures_rho)
        minradii, maxradii = calc_bands(masses, radii)
        numpy.save(root_name + 'minpres_rho', minpres_rho)
        numpy.save(root_name + 'maxpres_rho', maxpres_rho)
        numpy.save(root_name + 'minpres', minpres)
        numpy.save(root_name + 'maxpres', maxpres)
        numpy.save(root_name + 'minradii', minradii)
        numpy.save(root_name + 'maxradii', maxradii)


def cornerplot(root_name, variable_params):
    ewposterior = numpy.loadtxt(root_name + 'post_equal_weights.dat')
    figure = corner.corner(ewposterior[:,0:-1], labels = list(variable_params.keys()), show_titles=True, 
                      color=colors[4], quantiles =[0.16, 0.5, 0.84], smooth=.8)
    figure.savefig(root_name + 'corner.png')

def mass_radius_posterior_plot(root_name):
    scatter = numpy.load(root_name + 'scattered.npy')
    figure, ax = pyplot.subplots(1,1, figsize=(9,6))
    M_max = 0.
    for i in range(len(scatter[0])):
        corner.hist2d(scatter[:,i][:,3], scatter[:,i][:,2], labels = ['R [km]', 'M [M$_{\odot}$]'], show_titles=True, 
                        color=colors[i], smooth=.8, data_kwargs={'ms':5, 'alpha':0.5})
        M_max = max(max(scatter[:,i][:,2]), M_max)

    ax.set_xlim(8, 15)
    ax.set_ylim(1., M_max)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'Radius [km]', fontsize=15)
    ax.set_ylabel(r'Mass [M$_{\odot}$]', fontsize=15)
    pyplot.tight_layout()
    pyplot.show()
    figure.savefig(root_name + 'MRposterior.png')

def mass_radius_prior_predictive_plot(root_name,variable_params, label_name='updated prior', prior_mrpredictive=None):
    fig, ax = pyplot.subplots(1,1, figsize=(9, 6))

    num_stars = len(numpy.array([v for k,v in variable_params.items() if 'rhoc' in k]))

    if len(list(variable_params.keys())) == num_stars:
        flag = True

    else: 
        flag = False

    if flag == True:
        raise Exception("Cannot perform mass_radius_prior_predictive_plot function because EoS is fixed, i.e., tabulated or all EoS params are static params!")
    else:
        if prior_mrpredictive is not None:
            MRprior = numpy.load(prior_mrpredictive) 
            inbins = numpy.histogramdd(MRprior, bins=50, normed=True)
            levels = getdist.densities.getContourLevels(inbins[0], contours=[0.95])
            sns.kdeplot(x= MRprior[:,1], y=  MRprior[:,0], gridsize=50, 
                        shade=False, ax=ax, n_levels=levels[::-1], linewidths=2,
                        alpha=1., cmap=None, colors='black', linestyles='--')

        MR_prpr= numpy.loadtxt(root_name + 'MR_prpr.txt')
        inbins = numpy.histogramdd(MR_prpr[:,[1,0]], bins=50, normed=True)
        levels = getdist.densities.getContourLevels(inbins[0], contours=[0.68, 0.95])
    
        sns.kdeplot(x = MR_prpr[:,1], y = MR_prpr[:,0], gridsize=50, 
                    shade=True, ax=ax, n_levels=numpy.array([levels[1],levels[0], 1.]),
                    alpha=1., cmap=None, colors=colors[[5,3]])

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
        pyplot.tight_layout()
        fig.savefig(root_name + 'MRpriorpredictive.png')

def eos_posterior_plot(root_name,variable_params, prior_contours=None):
    fig, ax = pyplot.subplots(1,1, figsize=(9, 6))
    my_fontsize=20

    num_stars = len(numpy.array([v for k,v in variable_params.items() if 'rhoc' in k]))

    if len(list(variable_params.keys())) == num_stars:
        flag = True

    else: 
        flag = False

    if flag == True:
        raise Exception("Cannot perform mass_radius_prior_predictive_plot function because EoS is fixed, i.e., tabulated or all EoS params are static params!")
    else:
        minpres_pp = numpy.log10(numpy.load(root_name + 'minpres.npy'))
        maxpres_pp = numpy.log10(numpy.load(root_name + 'maxpres.npy'))
        scatter = numpy.load(root_name + 'scattered.npy')
        central_density_post = numpy.log10(scatter[:,0][:,[0,1]])

        corner.hist2d(central_density_post[:,0], central_density_post[:,1], show_titles=False, 
                            color=colors[3], plot_data_points=False, plot_density=False,
                    levels=[0.68, 0.95])


        ax.fill_between(minpres_pp[:,0], minpres_pp[:,2], maxpres_pp[:,2], 
                            color=sns.cubehelix_palette(8, start=.5, rot=-.75, dark=.2, light=.85)[0], alpha=1)
        ax.fill_between(minpres_pp[:,0], minpres_pp[:,1], maxpres_pp[:,1], 
                            color=sns.cubehelix_palette(8, start=.5, rot=-.75, dark=.2, light=.85)[3], alpha=1)
        if prior_contours is not None:
            minpres_prior = numpy.log10(numpy.load(prior_contours))
            maxpres_prior = numpy.log10(numpy.load(prior_contours))

            ax.plot(maxpres_prior[:,0], minpres_prior[:,2], c='black', linestyle='--', lw=2)
            ax.plot(maxpres_prior[:,0], maxpres_prior[:,2], c='black', linestyle='--', lw=2)

        ax.set_ylabel(r'$\log_{10}(P)$ (dyn/cm$^2$)', fontsize=my_fontsize)
        ax.set_xlabel(r'$\log_{10}(\varepsilon)$ (g/cm$^3$)', fontsize=my_fontsize)
        ax.tick_params(top=1,right=1, which='both', direction='in', labelsize=my_fontsize)

        ax.set_xlim(14.25, 15.24)
        ax.set_ylim(33, 36.2)

        pyplot.tight_layout()
        fig.savefig(root_name + 'EoSposterior.png')




