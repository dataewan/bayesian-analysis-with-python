import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")

n_params = [1, 2, 4]
p_params = [0.25, 0.5, 0.75]

x = np.arange(0, max(n_params) + 1)

fig, ax = plt.subplots(len(n_params), len(p_params), sharex=True, sharey=True)

for i in range(len(n_params)):
    for j in range(len(p_params)):
        n = n_params[i]
        p = p_params[j]

        y = stats.binom(n=n, p=p).pmf(x)

        ax[i, j].vlines(x, 0, y, colors="b", lw=5)
        ax[i, j].set_ylim(0, 1)
        ax[i, j].plot(0, 0, label="n = {:3.2f}\np={:3.2f}".format(n, p), alpha=0)
        ax[i, j].legend(fontsize=12)

ax[2, 1].set_xlabel(r"$\theta$", fontsize=14)
ax[1, 0].set_ylabel(r"$p(y|\theta)$", fontsize=14)
ax[0, 0].set_xticks(x)


plt.tight_layout()

plt.savefig("binomial_distribution.png")
