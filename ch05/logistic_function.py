import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)
logistic = 1 / (1 + np.exp(-z))

plt.plot(z, logistic)

plt.xlabel("$z$")
plt.ylabel("$logistic(z)$")

plt.savefig("logistic_function.png")
plt.close()
