import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pymc3 as pm
from scipy import stats

tips = sns.load_dataset("tips")

y = tips["tip"].values
idx = pd.Categorical(tips["day"]).codes

with pm.Model() as comparing_groups:
    means = pm.Normal("means", mu=0, sd=10, shape=len(set(idx)))
    sds = pm.HalfNormal("sds", sd=10, shape=len(set(idx)))

    y = pm.Normal("y", mu=means[idx], sd=sds[idx], observed=y)

    trace_cg = pm.sample(5000)

chain_cg = trace_cg[100::]
traceplot = pm.traceplot(chain_cg)

dist = stats.norm()

_, ax = plt.subplots(3, 2, figsize=(16, 12))
comparisons = [(i, j) for i in range(4) for j in range(i + 1, 4)]
pos = [(k, l) for k in range(3) for l in (0, 1)]

for (i, j), (k, l) in zip(comparisons, pos):
    means_diff = chain_cg["means"][:, i] - chain_cg["means"][:, j]
    d_cohen = (
        means_diff
        / np.sqrt((chain_cg["sds"][:, i] ** 2 + chain_cg["sds"][:, j] ** 2) / 2)
    ).mean()
    ps = dist.cdf(d_cohen / (2 ** 0.5))

    pm.plot_posterior(means_diff, ref_val=0, ax=ax[k, l], color="skyblue")

    ax[k, l].plot(
        0, label="Cohen's d = {:.2f}\nProb sup = {:.2f}".format(d_cohen, ps), alpha=0
    )

    ax[k, l].set_xlabel("$\mu_{} = \mu_{}$".format(i, j), fontsize=18)
    ax[k, l].legend(loc=0, fontsize=14)
    
plt.show()
