import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

N = 100
r = 0.8
x_0 = np.random.normal(size=N)
x_1 = np.random.normal(loc=x_0 * r, scale=(1 - r ** 2) ** 0.5)
y = np.random.normal(loc=x_0 - x_1)


def scatter_plot(x, y):
    plt.figure(figsize=(10, 10))

    for idx, x_i in enumerate(x):
        plt.subplot(2, 2, idx + 1)
        plt.scatter(x_i, y)
        plt.xlabel("$x_{}$".format(idx))
        plt.ylabel("$y$", rotation=0)

    plt.subplot(2, 2, idx + 2)
    plt.scatter(x[0], x[1])

    plt.xlabel("$x_{}$".format(idx - 1))
    plt.xlabel("$x_{}$".format(idx), rotation=0)


X = np.vstack((x_0, x_1))
scatter_plot(X, y)
plt.savefig("masking_effect_variables_data.png")
plt.close()

with pm.Model() as model_ma:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=10, shape=2)
    epsilon = pm.HalfCauchy("epsilon", 5)

    mu = alpha + pm.math.dot(beta, X)

    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=y)

    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_ma = pm.sample(5000, step=step, start=start)

pm.traceplot(trace_ma)
plt.savefig("masking_effect_variables_traceplot.png")
plt.close()

pm.forestplot(trace_ma, varnames=["beta"])
plt.savefig("masking_effect_variables_forestplot.png")
plt.close()
