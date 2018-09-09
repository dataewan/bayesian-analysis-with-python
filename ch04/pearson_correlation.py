import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pymc3 as pm

np.random.seed(314)

N = 100

alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x
y = y_real + eps_real

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1)
    epsilon = pm.HalfCauchy("epsilon", 5)

    mu = alpha + beta * x

    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=y)

    # calculate the pearson correlation coefficient by relating the slope and
    # the standard deviations of the variables
    rb = pm.Deterministic("rb", (beta * x.std() / y.std()) ** 2)

    # this is to do with least squares method, but we're skipping the details
    # of the derivation
    y_mean = y.mean()
    ss_reg = pm.math.sum((mu - y_mean) ** 2)
    ss_tot = pm.math.sum((y - y_mean) ** 2)

    rss = pm.Deterministic('rss', ss_reg / ss_tot)
    start = pm.find_MAP()
    step = pm.NUTS()
    trace_n = pm.sample(2000, step=step, start=start)

chain_n = trace_n[200:]
pm.traceplot(chain_n)

plt.savefig('pearson_correlation_traceplot.png')
plt.close()

print(pm.summary(chain_n))
