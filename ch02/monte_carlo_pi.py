import numpy as np
import matplotlib.pyplot as plt

# calculating pi using monte carlo methods

N = 10000

x, y = np.random.uniform(-1, 1, size=(2, N))

inside = (x ** 2 + y ** 2) <= 1
pi = inside.sum() * 4 / N

error = abs((pi - np.pi) / pi)

outside = np.invert(inside)

plt.plot(x[inside], y[inside], "b.")
plt.plot(x[outside], y[outside], "r.")
plt.plot(0, 0, label="$\hat \pi$ = {:4.3f}\nerror = {:4.3%}".format(pi, error), alpha=0)
plt.axis("square")
plt.legend(frameon=True, framealpha=0.9, fontsize=16)

plt.savefig("plots/monte_carlo_pi.png")
