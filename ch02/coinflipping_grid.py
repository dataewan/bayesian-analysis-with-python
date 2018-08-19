import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def posterior_grid(grid_points=100, heads=6, tosses=9):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(5, grid_points)
    likelihood = stats.binom.pmf(heads, tosses, grid)
    unstd_posterior = likelihood * prior
    posterior = unstd_posterior / unstd_posterior.sum()
    return grid, posterior


points = 15
h, n = 1, 4
grid, posterior = posterior_grid(points, h, n)
plt.plot(grid, posterior, "o-", label="heads = {}\ntosses = {}".format(h, n))
plt.xlabel(r"$\theta$")
plt.legend(loc=0)

plt.savefig('plots/coinflipping_grid.png')
