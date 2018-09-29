import seaborn as sns
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd

iris = sns.load_dataset("iris")

# Preprocessing the data
df = iris.query("species == ('setosa', 'versicolor')")
y_1 = pd.Categorical(df["species"]).codes
x_n = ["sepal_length", "sepal_width"]
x_1 = df[x_n].values

with pm.Model() as model_1:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=2, shape=len(x_n))

    mu = alpha + pm.math.dot(x_1, beta)

    # This is the effect of applying the logistic function to mu
    theta = pm.Deterministic("theta", 1 / (1 + pm.math.exp(-mu)))
    # This is the boundary decision, used to separate the classes
    bd = pm.Deterministic("bd", -alpha / beta[1] - beta[0] / beta[1] * x_1[:, 0])

    yl = pm.Bernoulli("yl", theta, observed=y_1)

    start = pm.find_MAP()
    step = pm.NUTS()
    trace_1 = pm.sample(5000, step, start)

varnames = ["alpha", "beta", "bd"]
pm.traceplot(trace_1, varnames)
plt.savefig("multiple_logistic_traceplot.png")
plt.close()

print(pm.summary(trace_1, varnames))

# plot the data along with the sigmoid curve

idx = np.argsort(x_1[:, 0])
bd = trace_1.bd.mean(0)[idx]
plt.scatter(x_1[:, 0], x_1[:, 1], c=y_1)
plt.plot(x_1[:, 0][idx], bd, color="r")

bd_hpd = pm.hpd(trace_1.bd)[idx]
plt.fill_between(x_1[:, 0][idx], bd_hpd[:, 0], bd_hpd[:, 1], color="r", alpha=0.5)

plt.xlabel(x_n[0])
plt.ylabel(x_n[1])

plt.savefig("multiple_logistic_fit.png")
plt.close()
