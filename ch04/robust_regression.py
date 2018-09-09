import seaborn as sns
import numpy as np
import pymc3 as pm
from matplotlib import pyplot as plt
from scipy import stats

ans = sns.load_dataset("anscombe")
x_3 = ans[ans.dataset == "III"]["x"].values
y_3 = ans[ans.dataset == "III"]["y"].values

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]
plt.plot(
    x_3,
    (alpha_c + beta_c * x_3),
    "k",
    label="y = {:.2f} + {:.2f} * x".format(alpha_c, beta_c),
)
plt.plot(x_3, y_3, "bo")
plt.xlabel("$x")
plt.ylabel("$y$", rotation=0)
plt.legend(loc=0)
plt.subplot(1, 2, 2)
sns.kdeplot(y_3)
plt.xlabel("$y$")

plt.savefig("robust_regression_data.png")
plt.close()

# rewriting the model as a robust regression with student's t distribution
with pm.Model() as model_t:
    alpha = pm.Normal("alpha", mu=0, sd=100)
    beta = pm.Normal("beta", mu=0, sd=1)
    epsilon = pm.HalfCauchy("epsilon", 5)
    nu = pm.Deterministic("nu", pm.Exponential("nu_", 1 / 29) + 1)

    y_pred = pm.StudentT(
        "y_pred", mu=alpha + beta * x_3, sd=epsilon, nu=nu, observed=y_3
    )

    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_t = pm.sample(2000, step=step, start=start)

pm.traceplot(trace_t)
plt.savefig("robust_regression_traceplot.png")
print(pm.summary(trace_t))
