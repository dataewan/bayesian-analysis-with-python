import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import numpy as np
import seaborn as sns

clusters = 3
n_cluster = [90, 50, 75]
n_total = sum(n_cluster)
means = [9, 21, 35]
std_devs = [2, 2, 2]

mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

sns.kdeplot(np.array(mix))
plt.xlabel("$x$")
plt.savefig("gaussian_mixture_data.png")
plt.close()


with pm.Model() as gm_model:
    p = pm.Dirichlet("p", a=np.ones(clusters))

    means = pm.Normal("means", mu=[10, 20, 35], sd=2, shape=clusters)

    sd = pm.HalfCauchy("sd", 5, shape=clusters)

    y = pm.NormalMixture("y", w=p, mu=means, sd=sd, observed=mix)

    trace = pm.sample(2000)

pm.traceplot(trace)
plt.savefig("gaussian_mixture_traceplot.png")
plt.close()
print(pm.summary(trace))

ppc = pm.sample_ppc(trace, 50, gm_model)

for i in ppc['y']:
    sns.kdeplot(i, alpha=0.1, color='b')

sns.kdeplot(np.array(mix), color='k')
plt.xlabel('$x$')
plt.savefig("gaussian_mixture_ppc.png")
plt.close()
