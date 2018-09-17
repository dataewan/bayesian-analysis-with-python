import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

np.random.seed(314)

N = 100
alpha_real = 2.5
beta_real = [0.9, 1.5]
eps_real = np.random.normal(0, 0.5, size=N)


# generate the data
X = np.array([np.random.normal(i, j, N) for i, j in zip([10, 2], [1, 1.5])])
X_mean = X.mean(axis=1, keepdims=True)
X_centred = X - X_mean
y = alpha_real + np.dot(beta_real, X) + eps_real

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


scatter_plot(X_centred, y)
plt.savefig("multipleregression_data.png")


# now create the model

with pm.Model() as model_mlr:
    alpha_tmp = pm.Normal("alpha_tmp", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1, shape=2)
    epsilon = pm.HalfCauchy("epsilon", 5)

    mu = alpha_tmp + pm.math.dot(beta, X_centred)

    alpha = pm.Deterministic("alpha", alpha_tmp - pm.math.dot(beta, X_mean))

    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=y)

    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_mlr = pm.sample(5000, step=step, start=start)

pm.traceplot(trace_mlr)
plt.savefig('multipleregression_traceplot.png')
plt.close()
print(pm.summary(trace_mlr))
