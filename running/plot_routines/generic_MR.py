# Generic script for plotting an M-R distribution.

# Standard libraries
import sys

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# File to plot
fname = sys.argv[1]
data = np.loadtxt(fname)

# Create figure
figsize = (4.8, 3.0)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)

# Plot
levels = [0.05, 0.32, 1] # Show 95% and 68% contours
x = data[:, 1]
y = data[:, 0]
sns.kdeplot(x=x, y=y, gridsize=40, bw_adjust=1.5, fill=True, ax=ax, levels=levels, alpha=1.0)

# Set labels
xlabel = '$R$ [km]'
ylabel = r'$M$ [$M_{\odot}$]'
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

# Save result
fig.savefig('MR.pdf', bbox_inches='tight')
