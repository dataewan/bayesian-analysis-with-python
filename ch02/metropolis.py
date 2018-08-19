from scipy import stats
import numpy as np
from matplotlib import pyplot as plt


def metropolis(func, steps=10000):
    samples = np.zeros(steps)
    old_x = func.mean()
    old_prob = func.pdf(old_x)

    for i in range(steps):
        new_x = old_x + np.random.normal(0, 0.5)
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob

        if acceptance >= np.random.random():
            samples[i] = new_x
            old_x = new_x
            old_prob = new_prob

        else:
            samples[i] = old_x

    return samples


func = stats.beta(0.4, 2)
samples = metropolis(func)

x = np.linspace(0.01, 0.99, 100)
y = func.pdf(x)

plt.xlim(0, 1)
plt.plot(x, y, "r-", lw=3, label="True distribution")
plt.hist(samples, bins=30, density=True, label="Estimated distribution")
plt.xlabel("$x$")
plt.ylabel("$pdf(x)$")
plt.legend(fontsize=14)


plt.savefig("plots/metropolis.png")
