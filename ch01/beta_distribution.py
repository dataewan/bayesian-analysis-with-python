import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")

params = [0.5, 1, 2, 3]
p_params = [0.25, 0.5, 0.75]

x = np.linspace(0, 1, 100)

fig, ax = plt.subplots(len(params), len(params), sharex=True, sharey=True)

for i in range(len(params)):
    for j in range(len(params)):
        a = params[i]
        b = params[j]

        y = stats.beta(a, b).pdf(x)

        ax[i, j].plot(x, y)
        ax[i, j].plot(
            0, 0, label="$\\alpha$ = {:3.2f}\n$\\beta$={:3.2f}".format(a, b), alpha=0
        )
        ax[i, j].legend(fontsize=12)

ax[2, 1].set_xlabel(r"$\theta$", fontsize=14)
ax[1, 0].set_ylabel(r"$p(\theta)$", fontsize=14)


plt.tight_layout()

plt.savefig("beta_distribution.png")
