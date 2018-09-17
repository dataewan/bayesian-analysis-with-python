import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

# we're going to create eight related data groups, including one with just one
# datapoint

N = 20
M = 8

idx = np.repeat(range(M - 1), N)
idx = np.append(idx, 7)

alpha_real = np.random.normal(2.5, 0.5, size=M)
beta_real = np.random.beta(60, 10, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))

y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real

j, k = 0, N
for i in range(M):
    plt.subplot(2, 4, i + 1)
    plt.scatter(x_m[j:k], y_m[j:k])
    plt.xlim(6, 15)
    plt.ylim(7, 17)

    j += N
    k += N

plt.tight_layout()
plt.savefig("hierarchical_linear_regression_data.png")
plt.close()

# center the data before sending it to the model
x_centered = x_m - x_m.mean()

#   # first going to fit a non-hierarchical model, as we've already done.
#   with pm.Model() as unpooled_model:
#       alpha_tmp = pm.Normal("alpha_tmp", mu=0, sd=10, shape=M)
#       beta = pm.Normal("beta", mu=0, sd=10, shape=M)
#       epsilon = pm.HalfCauchy("epsilon", 5)

#       nu = pm.Exponential("nu", 1 / 30)

#       y_pred = pm.StudentT(
#           "y_pred",
#           mu=alpha_tmp[idx] + beta[idx] * x_centered,
#           sd=epsilon,
#           nu=nu,
#           observed=y_m,
#       )

#       alpha = pm.Deterministic("alpha", alpha_tmp - beta * x_m.mean())

#       start = pm.find_MAP()
#       step = pm.NUTS(scaling=start)
#       trace_up = pm.sample(2000, step=step, start=start)

#   varnames = ["alpha", "beta", "epsilon", "nu"]
#   pm.traceplot(trace_up, varnames)
#   plt.savefig("hierarchical_linear_regression_unhierarchical_traceplot.png")


with pm.Model() as hierarchical_model:
    alpha_tmp_mu = pm.Normal("alpha_tmp_mu", mu=0, sd=10)
    alpha_tmp_sd = pm.HalfNormal("alpha_tmp_sd", 10)
    beta_mu = pm.Normal("beta_mu", mu=0, sd=10)
    beta_sd = pm.HalfNormal("beta_sd", sd=10)

    alpha_tmp = pm.Normal("alpha_tmp", mu=alpha_tmp_mu, sd=alpha_tmp_sd, shape=M)
    beta = pm.Normal("beta", mu=beta_mu, sd=beta_sd, shape=M)
    epsilon = pm.HalfCauchy("epsilon", 5)
    nu = pm.Exponential("nu", 1 / 30)

    y_pred = pm.StudentT(
        "y_pred",
        mu=alpha_tmp[idx] + beta[idx] * x_centered,
        sd=epsilon,
        nu=nu,
        observed=y_m,
    )

    alpha = pm.Deterministic("alpha", alpha_tmp - beta * x_m.mean())
    alpha_mu = pm.Deterministic("alpha_mu", alpha_tmp_mu - beta_mu * x_m.mean())
    alpha_sd = pm.Deterministic("alpha_sd", alpha_tmp_sd - beta_mu * x_m.mean())

    trace_hm = pm.sample(1000)

varnames = [
    "alpha",
    "alpha_mu",
    "alpha_sd",
    "beta",
    "beta_mu",
    "beta_sd",
    "epsilon",
    "nu",
]
pm.traceplot(trace_hm, varnames)
plt.savefig("hierarchical_linear_regression_hierarchical_traceplot.png")

j, k = 0, N
x_range = np.linspace(x_m.min(), x_m.max(), 10)

for i in range(M):
    plt.subplot(2, 4, i + 1)
    plt.scatter(x_m[j:k], y_m[j:k])

    alfa_m = trace_hm["alpha"][:, i].mean()
    beta_m = trace_hm["beta"][:, i].mean()
    plt.plot(x_range, alfa_m + beta_m * x_range, c='k', label="y = {:.2f} + {:.2f} $\\times x$".format(alfa_m, beta_m))
    plt.xlim(x_m.min() - 1, x_m.max() + 1)
    plt.ylim(y_m.min() - 1, y_m.max() + 1)

    j += N
    k += N

plt.savefig("hierarchical_linear_regression_fitted.png")
