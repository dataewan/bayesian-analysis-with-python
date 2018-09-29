import seaborn as sns
import pandas as pd
import numpy as np
import pymc3 as pm
from matplotlib import pyplot as plt
import theano.tensor as tt

iris = sns.load_dataset("iris")

y_s = pd.Categorical(iris.species).codes
x_n = iris.columns[:-1]
x_s = iris[x_n].values
x_s = (x_s - x_s.mean(axis=0)) / x_s.std(axis=0)


with pm.Model() as model_s:
    alpha = pm.Normal("alpha", mu=0, sd=2, shape=3)
    beta = pm.Normal("beta", mu=0, sd=2, shape=(4, 3))

    mu = alpha + pm.math.dot(x_s, beta)
    theta = tt.nnet.softmax(mu)

    yl = pm.Categorical("yl", p=theta, observed=y_s)
    start = pm.find_MAP()
    step = pm.NUTS()
    trace_s = pm.sample(2000, step, start)


pm.traceplot(trace_s)
plt.savefig("softmax_traceplot.png")
plt.close()
