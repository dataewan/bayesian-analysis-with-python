import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

np.random.seed(314)

N = 100

x_1 = np.random.normal(size=N)
x_2 = x_1 + np.random.normal(size=N, scale=N)
y = x_1 + np.random.normal(size=N)

X = np.vstack([x_1, x_2])

# now define a function to plot three scatterplots


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


scatter_plot(X, y)
plt.savefig("confounding_data.png")


# now create the model

with pm.Model() as model_mlr:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1, shape=2)
    # beta = pm.Normal("beta", mu=0, sd=1)
    epsilon = pm.HalfCauchy("epsilon", 5)

    mu = alpha + pm.math.dot(beta, X)
    # mu = alpha + beta * x_2

    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=y)

    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_red = pm.sample(5000, step=step, start=start)

pm.traceplot(trace_red)
plt.savefig("confounding_traceplot.png")
plt.close()
print(pm.summary(trace_red))


