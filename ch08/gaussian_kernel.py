import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

np.random.seed(1)
x = np.random.uniform(0, 10, size=20)
y = np.random.normal(np.sin(x), 0.2)

plt.plot(x, y, "o")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")

plt.savefig("gaussian_kernel_data.png")
plt.close()


def gauss_kernel(x, n_knots=5, w=2):
    """Simple Gaussian Kernel model"""
    knots = np.linspace(np.floor(x.min()), np.ceil(x.max()), n_knots)
    return np.array([np.exp(-(x - k) ** 2 / w) for k in knots])


n_knots = 5

with pm.Model() as kernel_model:
    gamma = pm.Cauchy("gamma", alpha=0, beta=1, shape=n_knots)
    sd = pm.Uniform("sd", 0, 10)
    mu = pm.math.dot(gamma, gauss_kernel(x, n_knots))
    yl = pm.Normal("yl", mu=mu, sd=sd, observed=y)

    trace = pm.sample(10000)

chain = trace[5000:]
pm.traceplot(chain)
plt.savefig("gaussian_kernel_traceplot.png")
plt.close()


# plot the posterior predictive check
ppc = pm.sample_ppc(chain, model=kernel_model, samples=100)
plt.plot(x, ppc["yl"].T, "ro", alpha=0.1)
plt.plot(x, y, "bo")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.savefig("gaussian_kernel_ppc.png")
plt.close()

# see how the model performs with new data points
new_x = np.linspace(np.floor(x.min()), np.ceil(x.max()), 100)
k = gauss_kernel(new_x, n_knots)
gamma_pred = chain.gamma

for i in range(100):
    idx = np.random.randint(0, len(gamma_pred))
    y_pred = np.dot(gamma_pred[idx], k)
    plt.plot(new_x, y_pred, "r-", alpha=0.1)

plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.plot(x, y, "bo")
plt.savefig("gaussian_kernel_ppc_newpoints.png")
plt.close()
