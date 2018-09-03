import numpy as np
import pymc3 as pm
from scipy import stats
from matplotlib import pyplot as plt

N_samples = [30, 30, 30]
G_samples = [18, 18, 18]

group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []

for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i] - G_samples[i]]))


with pm.Model() as model_h:
    alpha = pm.HalfCauchy('alpha', beta=10)
    beta = pm.HalfCauchy('beta', beta=10)

    theta = pm.Beta('theta', alpha, beta, shape=len(N_samples))

    y = pm.Bernoulli('y', p=theta[group_idx], observed=data)

    trace_h = pm.sample(2000)

chain_h = trace_h[200:]

# We may alos be interested in seeing what the estimated prior looks like. One
# way to do this is this:

x = np.linspace(0, 1, 100)
for i in np.random.randint(0, len(chain_h), size=100):
    pdf = stats.beta(chain_h['alpha'][i], chain_h['beta'][i]).pdf(x)
    plt.plot(x, pdf, 'g', alpha=0.05)

dist = stats.beta(chain_h['alpha'].mean(), chain_h['beta'].mean())

pdf = dist.pdf(x)
mode = x[np.argmax(pdf)]
mean = dist.moment(1)

plt.plot(x, pdf, label="mode = {:.2f}\nmean = {:.2f}".format(mode, mean))

plt.legend(fontsize=14)

plt.xlabel(r"$\theta_{prior}$")
plt.show()
