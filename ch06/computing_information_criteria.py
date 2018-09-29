import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm

real_alpha = 4.25
real_beta = [8.7, -1.2]
data_size = 20

noise = np.random.normal(0, 2, size=data_size)
x_1 = np.linspace(0, 5, data_size)
y_1 = real_alpha + real_beta[0] * x_1 + real_beta[1] * x_1 ** 2 + noise
order = 2

x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

plt.scatter(x_1s[0], y_1s)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig("computing_information_criteria_data.png")
plt.close()


with pm.Model() as model_1:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=10)
    epsilon = pm.HalfCauchy("epsilon", 5)
    mu = alpha + beta * x_1s[0]
    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=y_1s)

    trace_1 = pm.sample(2000)

chain_1 = trace_1[100:]

pm.traceplot(chain_1)
plt.savefig("cic_model1_traceplot.png")
plt.close()


with pm.Model() as model_2:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=10, shape=x_1s.shape[0])
    epsilon = pm.HalfCauchy("epsilon", 5)

    mu = alpha + pm.math.dot(beta, x_1s)

    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=y_1s)

    trace_2 = pm.sample(2000)

chain_2 = trace_2[100:]

pm.traceplot(chain_2)
plt.savefig("cic_model2_traceplot.png")
plt.close()


alpha_1_post = chain_1.alpha.mean()
betas_1_post = chain_1.beta.mean()
idx = np.argsort(x_1s[0])
y_1_post = alpha_1_post + betas_1_post * x_1s[0]
plt.plot(x_1s[0][idx], y_1_post[idx], label="Linear")

alpha_2_post = chain_2.alpha.mean()
betas_2_post = chain_2.beta.mean(axis=0)
y_2_post = alpha_2_post + np.dot(betas_2_post, x_1s)
plt.plot(x_1s[0][idx], y_2_post[idx], label="Polynomial")

plt.scatter(x_1s[0], y_1s)
plt.legend()

plt.savefig("cic_plotting_fitted_data.png")
plt.close()


# posterior predictive checks

plt.subplot(121)
plt.scatter(x_1s[0], y_1s, c="r")
plt.ylim(-3, 3)
plt.xlabel("x")
plt.ylabel("y", rotation=0)
plt.title("Linear")
for i in range(0, len(chain_1.alpha), 50):
    plt.scatter(
        x_1s[0],
        chain_1.alpha[i] + chain_1.beta[i] * x_1s[0],
        edgecolors="g",
        alpha=0.05,
    )
plt.plot(x_1s[0], chain_1.alpha.mean() + chain_1.beta.mean() * x_1s[0], c="g", alpha=1)


plt.subplot(122)
plt.scatter(x_1s[0], y_1s, c="r")
plt.ylim(-3, 3)
plt.xlabel("x")
plt.ylabel("y", rotation=0)
plt.title("Order {}".format(order))
for i in range(0, len(chain_2.alpha), 50):
    plt.scatter(
        x_1s[0],
        chain_2.alpha[i] + np.dot(chain_2.beta[i], x_1s),
        c="g",
        edgecolors="g",
        alpha=0.05,
    )

plt.plot(
    x_1s[0],
    alpha_2_post + np.dot(betas_2_post, x_1s),
    c="g",
    alpha=1,
)

plt.savefig("cic_posterior_predictive_checks.png")
plt.close()
