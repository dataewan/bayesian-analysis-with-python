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

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y, "b.")
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.plot(x, y_real, "k")
plt.subplot(1, 2, 2)
sns.kdeplot(y)
plt.xlabel("$y$", fontsize=16)

plt.savefig("linear_regression_data.png")
plt.close()


with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1)
    epsilon = pm.HalfCauchy("epsilon", 5)

    mu = pm.Deterministic("mu", alpha + beta * x)

    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=y)
    # Alternatively we could just write:
    # y_pred = pm.Normal("y_pred", mu=alpha + beta * x, sd=epsilon, observed=y)

    start = pm.find_MAP()
    step = pm.NUTS()
    trace = pm.sample(10000, step, start)

chain = trace[200:]
pm.traceplot(chain)

plt.savefig("linear_regression_traceplot.png")
plt.close()

varnames = ["alpha", "beta", "epsilon"]
pm.autocorrplot(trace, varnames)
plt.savefig("linear_regression_autocorrplot.png")
plt.close()


# Interpreting and visualising the posterior

# plotting the average line

plt.plot(x, y, "b.")
alpha_m = trace["alpha"].mean()
beta_m = trace["beta"].mean()

plt.plot(
    x,
    alpha_m + beta_m * x,
    c="k",
    label="y = {:.2f} + {:.2f} * x".format(alpha_m, beta_m),
)
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$y$", fontsize=16)
plt.legend(loc=2, fontsize=14)
plt.savefig("linear_regression_averageline.png")
plt.close()

# plotting uncertainty with semitransparent lines

plt.plot(x, y, "b.")
alpha_m = trace["alpha"].mean()
beta_m = trace["beta"].mean()

idx = range(0, len(trace["alpha"]), 10)
plt.plot(
    x, trace["alpha"][idx] + trace["beta"][idx] * x[:, np.newaxis], c="grey", alpha=0.5
)

plt.plot(
    x,
    alpha_m + beta_m * x,
    c="k",
    label="y = {:.2f} + {:.2f} * x".format(alpha_m, beta_m),
)
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$y$", fontsize=16)
plt.legend(loc=2, fontsize=14)
plt.savefig("linear_regression_semitransparentlines.png")
plt.close()

# plotting HPD

alpha_m = trace["alpha"].mean()
beta_m = trace["beta"].mean()
plt.plot(
    x,
    alpha_m + beta_m * x,
    c="k",
    label="y = {:.2f} + {:.2f} * x".format(alpha_m, beta_m),
)

idx = np.argsort(x)
x_ord = x[idx]
sig = pm.hpd(trace["mu"], alpha=0.02)[idx]
plt.fill_between(x_ord, sig[:, 0], sig[:, 1], color="grey")

plt.xlabel("$x$", fontsize=16)
plt.ylabel("$y$", fontsize=16)
plt.legend(loc=2, fontsize=14)
plt.savefig("linear_regression_hpd.png")
plt.close()


# taking posterior predictive samples

ppc = pm.sample_ppc(chain, samples=1000, model=model)

plt.plot(x, y, "b.")
alpha_m = trace["alpha"].mean()
beta_m = trace["beta"].mean()
plt.plot(
    x,
    alpha_m + beta_m * x,
    c="k",
    label="y = {:.2f} + {:.2f} * x".format(alpha_m, beta_m),
)

ppc = pm.sample_ppc(chain, samples=1000, model=model)
sig0 = pm.hpd(ppc['y_pred'], alpha=0.5)[idx]
sig1 = pm.hpd(ppc['y_pred'], alpha=0.05)[idx]

plt.fill_between(x_ord, sig0[:, 0], sig0[:, 1], color="grey")
plt.fill_between(x_ord, sig1[:, 0], sig1[:, 1], color="grey", alpha=0.5)

plt.xlabel("$x$", fontsize=16)
plt.ylabel("$y$", fontsize=16)
plt.legend(loc=2, fontsize=14)
plt.savefig("linear_regression_postpredictivesamples.png")
plt.close()
