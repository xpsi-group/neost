import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy import stats
from matplotlib import rcParams
from numpy.random import Generator

plt.rcdefaults()

plt.rcParams['xtick.direction']       = 'in'
plt.rcParams['xtick.minor.visible']   = True
plt.rcParams['ytick.direction']       = 'in'
plt.rcParams['ytick.minor.visible']   = True
plt.rcParams['ytick.right']           = True
plt.rcParams['xtick.top']             = True
plt.rcParams['xtick.major.pad']       = 6

my_fontsize = 12
rcParams.update({'font.size': my_fontsize})

kramer_input = np.loadtxt('data/pdf_Ip45_mp.dat')

# reshape to [mass, moi][n_samples] (personal preference)
kramer_temp = np.transpose(kramer_input, (1, 0))
kramer = np.zeros_like(kramer_temp)
kramer[0] = kramer_temp[1]
kramer[1] = kramer_temp[0]

mean_mass = round(np.mean(kramer[0]), 4)
print(f'Mean mass: {mean_mass}')
std_mass = np.std(kramer[0])
print(f'Standard deviation of mass: {std_mass}')

pdf = stats.norm(loc=mean_mass, scale=std_mass).pdf

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
sns.histplot(kramer[0] - 1.338, binrange=(0, 0.0005), bins=50, color='tab:blue', ax=axes[0], stat='density', label='Kramer et al. (2021)')
axes[0].set_xlabel('M - 1.338 [$M_\odot$]')
axes[0].plot(np.linspace(1.338, 1.3385, 1000) - 1.338, pdf(np.linspace(1.338, 1.3385, 1000)), color='k', label='Normal fit')
axes[0].set_xlim(0, 0.0005)
axes[0].set_ylim(0, 8000)
axes[0].legend(loc='upper right')

weights = pdf(kramer[0])
p = weights / np.sum(weights)
n_eff = 1 / np.sum(p**2)
ratio = n_eff / kramer.shape[1]
print(f'Effective sample size: {round(n_eff)} ({ratio:.2%} of original)')

# already saved and used in inference run
# kramer_resampled = np.random.choice(kramer[1], size=kramer.shape[1], replace=True, p=p)
kramer_resampled = np.load('runs/pdf_Ip45_mp_resampled.npy')

sns.histplot(kramer[1], binrange=(0, 8), bins=40, color='tab:blue', ax=axes[1], stat='density', label='Kramer et al. (2021)')
sns.histplot(kramer_resampled, binrange=(0, 8), bins=40, color='tab:red', ax=axes[1], stat='density', label='Resampled', alpha=0.8)
axes[1].legend(loc='upper right')
axes[1].set_ylim(0, 0.6)
axes[1].set_xlim(0, 6)
axes[1].set_xlabel('I [$10^{45}$ g cm$^2$]')

fig.savefig('Figures/kramer_resample.png')

print(kramer_resampled.shape)

# already saved and used in inference run
# np.save('runs/pdf_Ip45_mp_resampled.npy', kramer_resampled)