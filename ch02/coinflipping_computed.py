import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pymc3 as pm

np.random.seed(123)

n_experiments = 40
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)
print(data)

with pm.Model() as our_first_model:
    theta = pm.Beta("theta", alpha=1, beta=1)
    y = pm.Bernoulli("y", p=theta, observed=data)

    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step, start=start)

    burnin = 100
    chain = trace[burnin:]
    pm.traceplot(chain, lines={"theta": theta_real})


plt.savefig("plots/coinflipping_computed.png")
plt.close()

with our_first_model:
    step = pm.Metropolis()
    multi_trace = pm.sample(1000, step=step, chains=4)

burnin = 0
multichain = multi_trace[burnin:]
pm.traceplot(multichain, lines={"theta": theta_real})

plt.savefig("plots/coinflipping_computed_multitrace.png")
plt.close()
