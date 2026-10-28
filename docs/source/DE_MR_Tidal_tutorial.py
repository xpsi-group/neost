#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import time
import neost
from neost.eos import polytropes
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from scipy.stats import multivariate_normal
from neost import PosteriorAnalysis
import numpy as np
import matplotlib.pyplot as plt
import timeit
import neost.global_imports as global_imports
from matplotlib.patches import Ellipse

start = time.time()
c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons

print(np.log10(36.864753138526*rho_ns))

EOS = polytropes.PolytropicEoS(crust = 'ceft-Keller-N3LO', rho_t = 1.1*rho_ns,adm_type = 'Dark Energy')


EOS.update({'gamma1':2.3, 'gamma2':4., 'gamma3':2.6, 'rho_t1':1.8, 'rho_t2':4, 'A_param': 0.3, 'alpha': 0.4, 'rho_plus': 2.78, 'ceft': 2.6}, max_edsc_de = True)


central_densities = np.linspace(EOS.rho_plus/rho_ns + 0.1, EOS.max_edsc_de, 75)*rho_ns


MR = np.zeros((len(central_densities), 6))



print(EOS.max_edsc_de, EOS.max_edsc/rho_ns)

for i, eps in enumerate(central_densities):
    star = Star(eps,0.0,EOS.rho_plus, EOS.alpha, False, True)
    star.solve_structure(EOS.energydensities, EOS.pressures, EOS.energydensities_de, EOS.pressures_de)

    MR[i] = star.Mrot, star.Req,star.tidal, 0.0,0.0,star.Mdm


end = time.time()

print(MR[:,0])
print(MR[:,1])
print("Execution time of the MR is: " + str(end-start)) 



fig, ax = plt.subplots(1,1, figsize=(10, 6))
lns1 = ax.plot(MR[:,1], MR[:,0],label='DE neutron star', lw=2.5,color = '#00B1B7')
lns2 = ax.plot(EOS.massradius[:,1], EOS.massradius[:,0], label='Baryonic neutron star', lw=2.5,alpha = 0.5,color = '#005ABD')
ax.set_ylim(0.,3.)
ax.set_xlim(8,16)


ax.set_xlabel(r'Radius [km]', fontsize=18)
ax.set_ylabel(r'Mass [M$_\odot$]', fontsize=18)
ax.tick_params(width=2, labelsize=12, direction='in')
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc='best', fontsize=16)

plt.tight_layout()
plt.show()
plt.savefig('chains/' + 'MR_DE_example.png')