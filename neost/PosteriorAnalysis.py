import numpy as np
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
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

colors = np.array(["#c6878f", "#b79d94", "#969696", "#67697c", "#233b57", "#BCBEC7"])

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
        contours = np.nanquantile(array, quantiles) #changed to nanquantile to inorder to ignore the nans that may appear
        low = contours[0]
        median = contours[1]
        high = contours[2]
        minus = low - median
        plus = high - median
        return np.round(median,2),np.round(plus,2),np.round(minus,2)  #returns uncertainties on the array

def compute_table_data(root_name, EOS, variable_params, static_params):
    ewposterior = np.loadtxt(root_name + 'post_equal_weights.dat')
    print("Total number of samples is %d" %(len(ewposterior)))
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
        Data_array = np.zeros((len(ewposterior),13)) #contains Mtov, Rtov, eps_cent TOV, rho_cent TOV, P_cent TOV,R 1.4, eps_cent 1.4, rho_cent 1.4,
                                                                #P_cent 1.4, R 2.0, eps_cent 2.0, rho_cent 2.0, P_cent 2.0

        for i in range(0, len(ewposterior), 1):
            pr = ewposterior[i][0:len(variable_params)]
            par = {e:pr[j] for j, e in enumerate(list(variable_params.keys()))}
            par.update(static_params)
            EOS.update(par, max_edsc=True)

            edsrho = UnivariateSpline(EOS.energydensities, EOS.massdensities, k=1, s=0)
            max_rhoc = edsrho(EOS.max_edsc) / rho_ns #division by rho_ns gives max_rhoc in terms of n_c/n_0 as mass density and number density only differ by a factor the mass of baryon, which is canceled out in this fraction

            eps = np.logspace(14.4, np.log10(EOS.max_edsc), 40)
            M = np.zeros(len(eps))
            R = np.zeros(len(eps))
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
                rho_14 = edsrho(eps_14) / rho_ns
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

def compute_prior_auxiliary_data(root_name, EOS, variable_params, static_params):
    ewprior = np.loadtxt(root_name + 'post_equal_weights.dat')
    print("Total number of samples is %d" %(len(ewprior)))

    num_stars = len(np.array([v for k,v in variable_params.items() if 'rhoc' in k]))

    if len(list(variable_params.keys())) == num_stars:
        flag = True

    else:
        flag = False

    masses = np.linspace(.2, 2.9, 50)
    energydensities = np.logspace(14.2, 16, 50)

    if flag == True:
        pressures = np.zeros((len(masses), len(ewprior)))
        pressures_rho = np.zeros((len(masses), len(ewprior)))
        MR_prpr_pp = np.zeros((len(ewprior), 2))
    else:
        pressures = np.zeros((len(masses), len(ewprior)))
        pressures_rho = np.zeros((len(masses), len(ewprior)))
        minpres = np.zeros((3, len(energydensities)))
        maxpres = np.zeros((3, len(energydensities)))
        minpres_rho = np.zeros((3, len(energydensities)))
        maxpres_rho = np.zeros((3, len(energydensities)))
        MR_prpr_pp = np.zeros((len(ewprior), 2))

    for i in range(0, len(ewprior), 1):

        pr = ewprior[i][0:len(variable_params)]
        par = {e:pr[j] for j, e in enumerate(list(variable_params.keys()))}
        par.update(static_params)
        EOS.update(par, max_edsc=True)

        rhopres = UnivariateSpline(EOS.massdensities, EOS.pressures, k=1, s=0)
        edsrho = UnivariateSpline(EOS.energydensities, EOS.massdensities, k=1, s=0)
        max_rhoc = edsrho(EOS.max_edsc)
        pressures_rho[:,i][energydensities<max_rhoc] = rhopres(energydensities[energydensities<max_rhoc])
        pressures[:,i][energydensities<EOS.max_edsc] = EOS.eos(energydensities[energydensities<EOS.max_edsc])

        rhoc = 10**par['rhoc_1'] #just pick one of the central density samples, their distributions will be identical since constant likelihood eval on all sources
        star = Star(rhoc)
        star.solve_structure(EOS.energydensities, EOS.pressures)
        MR_prpr_pp[i] = star.Mrot, star.Req
    # save everything
    np.save(root_name + 'pressures', pressures)
    np.savetxt(root_name + 'MR_prpr.txt', MR_prpr_pp)

    if flag == False:
        minpres, maxpres = calc_bands(energydensities, pressures)
        minpres_rho, maxpres_rho = calc_bands(energydensities, pressures_rho)
        np.save(root_name + 'minpres_rho', minpres_rho)
        np.save(root_name + 'maxpres_rho', maxpres_rho)
        np.save(root_name + 'minpres', minpres)
        np.save(root_name + 'maxpres', maxpres)


def compute_auxiliary_data(root_name, EOS, variable_params, static_params, chirp_masses): 
    ewposterior = np.loadtxt(root_name + 'post_equal_weights.dat')
    print("Total number of samples is %d" %(len(ewposterior)))

    num_stars = len(np.array([v for k,v in variable_params.items() if 'rhoc' in k]))

    if len(list(variable_params.keys())) == num_stars:
        flag = True

    else: 
        flag = False

    masses = np.linspace(.2, 2.9, 50)
    energydensities = np.logspace(14.2, 16, 50)
    scattered = []

    if flag == True:
        radii = np.zeros((len(masses), len(ewposterior)))
        pressures = np.zeros((len(masses), len(ewposterior)))
        pressures_rho = np.zeros((len(masses), len(ewposterior)))
        MR_prpr_pp = np.zeros((len(ewposterior), 2))

    else:
        radii = np.zeros((len(masses), len(ewposterior)))
        pressures = np.zeros((len(masses), len(ewposterior)))
        pressures_rho = np.zeros((len(masses), len(ewposterior)))
        minradii = np.zeros((3, len(masses)))
        maxradii = np.zeros((3, len(masses)))
        minpres = np.zeros((3, len(energydensities)))
        maxpres = np.zeros((3, len(energydensities)))
        minpres_rho = np.zeros((3, len(energydensities)))
        maxpres_rho = np.zeros((3, len(energydensities)))
        MR_prpr_pp = np.zeros((len(ewposterior), 2))

    
    


    for i in range(0, len(ewposterior), 1):

        pr = ewposterior[i][0:len(variable_params)]
        par = {e:pr[j] for j, e in enumerate(list(variable_params.keys()))}
        par.update(static_params)
        EOS.update(par, max_edsc=True)

        rhopres = UnivariateSpline(EOS.massdensities, EOS.pressures, k=1, s=0)
        edsrho = UnivariateSpline(EOS.energydensities, EOS.massdensities, k=1, s=0)
        max_rhoc = edsrho(EOS.max_edsc)
        pressures_rho[:,i][energydensities<max_rhoc] = rhopres(energydensities[energydensities<max_rhoc])
        pressures[:,i][energydensities<EOS.max_edsc] = EOS.eos(energydensities[energydensities<EOS.max_edsc])
            
        rhocs = np.logspace(14.5, np.log10(EOS.max_edsc), 30)
        M = np.zeros(len(rhocs))
        R = np.zeros(len(rhocs))
        for j, e in enumerate(rhocs):
            star = Star(e)
            star.solve_structure(EOS.energydensities, EOS.pressures)
            M[j] = star.Mrot
            R[j] = star.Req

        M, indices = np.unique(M, return_index=True)
        MR = UnivariateSpline(M, R[indices], k=1, s=0, ext=1)
        rhocM = UnivariateSpline(M, rhocs[indices], k=1, s=0)
            
        rhocpar = np.array([10**v for k,v in par.items() if 'rhoc' in k])
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
        rhoc = np.random.rand() *(np.log10(EOS.max_edsc) - 14.6) + 14.6
        star = Star(10**rhoc)
        star.solve_structure(EOS.energydensities, EOS.pressures)
        MR_prpr_pp[i] = star.Mrot, star.Req

        radii[:,i] = MR(masses)


    scattered = np.array(scattered)
    # save everything
    np.save(root_name + 'pressures', pressures)
    np.save(root_name + 'radii', radii)
    np.save(root_name + 'scattered', scattered)
    np.savetxt(root_name + 'MR_prpr.txt', MR_prpr_pp)
   
    if flag == False:
        minpres, maxpres = calc_bands(energydensities, pressures)
        minpres_rho, maxpres_rho = calc_bands(energydensities, pressures_rho)
        minradii, maxradii = calc_bands(masses, radii)
        np.save(root_name + 'minpres_rho', minpres_rho)
        np.save(root_name + 'maxpres_rho', maxpres_rho)
        np.save(root_name + 'minpres', minpres)
        np.save(root_name + 'maxpres', maxpres)
        np.save(root_name + 'minradii', minradii)
        np.save(root_name + 'maxradii', maxradii)


def cornerplot(root_name, variable_params):
    ewposterior = np.loadtxt(root_name + 'post_equal_weights.dat')
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

def eos_posterior_plot(root_name,variable_params, prior_contours=None):
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
        minpres_pp = np.log10(np.load(root_name + 'minpres.npy'))
        maxpres_pp = np.log10(np.load(root_name + 'maxpres.npy'))
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

        ax.set_xlim(14.25, 15.24)
        ax.set_ylim(33, 36.2)

        plt.tight_layout()
        fig.savefig(root_name + 'EoSposterior.png')




