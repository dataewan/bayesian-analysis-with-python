import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymc3 as pm


sigma_x1 = 1
sigmas_x2 = [1, 2]

rhos = [-0.99, -0.5, 0, 0.5, 0.99]

x, y = np.mgrid[-5:5:0.1, -5:5:0.1]

pos = np.empty(x.shape + (2,))

pos[:, :, 0] = x
pos[:, :, 1] = y

f, ax = plt.subplots(
    len(sigmas_x2), len(rhos), sharex=True, sharey=True, figsize=(20, 10)
)

for i in range(len(sigmas_x2)):
    for j in range(len(rhos)):
        sigma_x2 = sigmas_x2[i]
        rho = rhos[j]

        # covariance matrix. 2x2, on the diagonal are the variances each
        # variable. The rest of the elements are the covariances expressed in
        # terms of the standard deviations and pearson's correlation
        # coefficient.
        cov = [
            [sigma_x1 ** 2, sigma_x1 * sigma_x2 * rho],
            [sigma_x1 * sigma_x2 * rho, sigma_x2 ** 2],
        ]

        rv = stats.multivariate_normal([0, 0], cov)

        ax[i, j].contour(x, y, rv.pdf(pos))
        ax[i, j].plot(
            0,
            0,
            label="$\\sigma_{{x2}}$ = {:3.2f}\n$\\rho$ = {:3.2f}".format(sigma_x2, rho),
            alpha=0,
        )
        ax[i, j].legend()

ax[1, 2].set_xlabel("$x_1$")
ax[1, 0].set_ylabel("$x_2$")

plt.savefig("multivariate_gaussians_contourplots.png")
plt.close()

np.random.seed(1)
N = 100
alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x
y = y_real + eps_real

data = np.stack((x, y)).T

with pm.Model() as model:
    mu = pm.Normal("mu", mu=data.mean(0), sd=10, shape=2)

    sigma_1 = pm.HalfNormal("sigma_1", 10)
    sigma_2 = pm.HalfNormal("sigma_2", 10)
    rho = pm.Uniform("rho", -1, 1)

    cov = pm.math.stack(
        (
            [sigma_1 ** 2, sigma_1 * sigma_2 * rho],
            [sigma_1 * sigma_2 * rho, sigma_2 ** 2],
        )
    )

    y_pred = pm.MvNormal("y_pred", mu=mu, cov=cov, observed=data)

    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_p = pm.sample(1000, step=step, start=start)

pm.traceplot(trace_p)
plt.savefig("multivariate_gaussians_traceplot.png", dpi=300, figsize=(5.5, 5.5))
plt.close()

print(pm.summary(trace_p))
