import os
import requests
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

FISHFILE = "fish.csv"
FISHURL = "https://stats.idre.ucla.edu/stat/data/fish.csv"

if not os.path.exists(FISHFILE):
    r = requests.get(FISHURL)
    with open(FISHFILE, "wb") as f:
        f.write(r.content)

df = pd.read_csv(FISHFILE)
# This dataset includes data collected from a survey of 250 visitors who
# visited the park. The group level data consists of:
#  - The number of fish they caught (count)
#  - The number of children in the group (child)
#  - If they took a camper to the park (camper)

with pm.Model() as ZIP_reg:
    psi = pm.Beta("psi", 1, 1)

    alpha = pm.Normal("alpha", 0, 10)
    beta = pm.Normal("beta", 0, 10, shape=2)

    lam = pm.math.exp(alpha + beta[0] * df["child"] + beta[1] * df["camper"])

    y = pm.ZeroInflatedPoisson("y", theta=lam, psi=psi, observed=df["count"])

    trace_ZIP_reg = pm.sample(2000)


chain_ZIP_reg = trace_ZIP_reg[100:]
pm.traceplot(chain_ZIP_reg)
plt.savefig("fish_traceplot.png")
plt.close()


children = [0, 1, 2, 3, 4]
fish_count_pred_0 = []
fish_count_pred_1 = []
thin = 5

for n in children:
    without_camper = chain_ZIP_reg.alpha[::thin] + chain_ZIP_reg.beta[:, 0][::thin] * n

    with_camper = without_camper + chain_ZIP_reg.beta[:, 1][::thin]

    fish_count_pred_0.append(np.exp(without_camper))
    fish_count_pred_1.append(np.exp(without_camper))


plt.plot(children, fish_count_pred_0, "bo", alpha=0.01)
plt.plot(children, fish_count_pred_1, "ro", alpha=0.01)

plt.xticks(children)

plt.xlabel("Number of children")
plt.ylabel("Fish caught")

plt.plot([], 'bo', label='without camper')
plt.plot([], 'ro', label='with camper')

plt.savefig("fish_fit.png")
