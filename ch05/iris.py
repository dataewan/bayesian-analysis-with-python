import seaborn as sns
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd

iris = sns.load_dataset("iris")

sns.stripplot("species", "sepal_length", data=iris, jitter=True)
plt.savefig("iris_stripplot.png")
plt.close()

sns.pairplot(iris, hue="species", diag_kind="kde")
plt.savefig("iris_pairplot.png")
plt.close()

df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df["species"]).codes
x_n = "sepal_length"
x_0 = df[x_n].values

with pm.Model() as model_0:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=10)

    mu = alpha + pm.math.dot(x_0, beta)

    # This is the effect of applying the logistic function to mu
    theta = pm.Deterministic("theta", 1 / (1 + pm.math.exp(-mu)))
    # This is the boundary decision, used to separate the classes
    bd = pm.Deterministic("bd", -alpha / beta)

    yl = pm.Bernoulli("yl", theta, observed=y_0)

    start = pm.find_MAP()
    step = pm.NUTS()
    trace_0 = pm.sample(5000, step, start)

varnames = ["alpha", "beta", "bd"]
chain_0 = trace_0[1000:]
pm.traceplot(chain_0, varnames)
plt.savefig("logistic_iris_traceplot.png")
plt.close()

print(pm.summary(chain_0, varnames))
