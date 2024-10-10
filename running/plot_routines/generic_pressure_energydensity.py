# Generic script for plotting pressure vs energydensity.

# Stardard libraries
import sys

# 3rd party
import matplotlib.pyplot as plt
import numpy as np

# Local
import plotting

# Plotting function
plot = plotting.plot_1D_quantiles

# File to plot
fname = sys.argv[1]
data = np.load(fname)

# Create figure
figsize = (4.8, 3.0)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)

# Plot
print('N.B.: this script assumes that the x-values are defined as np.log10(np.logspace(14.2,16,50))')
colors = ['xkcd:lavender', 'xkcd:wisteria']
plot(ax, data, colors, 'filled')

# Set labels and titles
xlabel = r'$\log_{10}(\varepsilon)$ [g/cm$^3$]'
ylabel = r'$\log_{10}(P)$ [dyn/cm$^2$]'
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

# Save result
fig.savefig('P_vs_e.pdf', bbox_inches='tight')
