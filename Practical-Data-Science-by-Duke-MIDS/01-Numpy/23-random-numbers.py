import numpy as np
import matplotlib.pyplot as plt


#================================#
# Random numbers #
#================================#

# Uniformly distributed numbers between 0 and 1
a = np.random.rand(4)
print(a)

# Matrix of random numbers
a = np.random.rand(4, 4)
print(a)

# Gaussian
b = np.random.randn(4)
print(b)

# Generate an array of 10,000 random normal numbers
random_array = np.random.randn(10000)
print(random_array)

# Output a histogram of the data with 10 bins
bins = np.arange(-3.5, 3.5, 0.25)
hist, bin_edges = np.histogram(random_array, bins=bins)

# Plot the results:
plt.bar(bin_edges[:-1] + 0.125, hist)
plt.xlabel("Value")
plt.ylabel("Count")
plt.show()


#================================#
# Reproducible randomness #
#================================#

# Gaussian
b = np.random.randn(4)
print(b)

# Gaussian
c = np.random.randn(4)
print(c)

# Setting the random seed
np.random.seed(42)
b = np.random.randn(4)
print(b)

# If one runs it again (setting the same random seed), one gets identical results.

# Setting the random seed
np.random.seed(42)
# Gaussian
b = np.random.randn(4)
print(b)


#================================#
# Random sampling #
#================================#

# ID numbers for each member of the population we wish to sample from
ids = np.arange(20)
print(ids)

ids_shuffle = ids.copy()
np.random.shuffle(ids_shuffle)
print(ids_shuffle)

control = ids_shuffle[0:10]
experiment = ids_shuffle[10:]
print(control)
print(experiment)

