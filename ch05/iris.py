import seaborn as sns
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

sns.stripplot("species", "sepal_length", data=iris, jitter=True)
plt.savefig("iris_stripplot.png")
plt.close()

sns.pairplot(iris, hue="species", diag_kind="kde")
plt.savefig("iris_pairplot.png")
plt.close()
