import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm

ans = sns.load_dataset("anscombe")
x_2 = ans[ans.dataset == "II"]["x"].values
y_2 = ans[ans.dataset == "II"]["y"].values


x_2 = x_2 - x_2.mean()
y_2 = y_2 - y_2.mean()

plt.scatter(x_2, y_2)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig("polynomial_data.png")
plt.close()

with pm.Model() as model_poly:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta1 = pm.Normal("beta1", mu=0, sd=1)
    beta2 = pm.Normal("beta2", mu=0, sd=1)
    epsilon = pm.Uniform("epsilon", lower=0, upper=10)

    mu = alpha + beta1 * x_2 + beta2 * x_2 ** 2

    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=y_2)

    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_poly = pm.sample(3000, step=step, start=start)

pm.traceplot(trace_poly)
plt.savefig("polynomial_traceplot.png")
plt.close()

print(pm.summary(trace_poly))

x_p = np.linspace(-6, 6)
a = trace_poly["alpha"].mean()
b1 = trace_poly["beta1"].mean()
b2 = trace_poly["beta2"].mean()

y_p = a + b1 * x_p + b2 * x_p ** 2

plt.scatter(x_2, y_2)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.plot(x_p, y_p, c="r")
plt.savefig('polynomial_prediction.png')
plt.close()
