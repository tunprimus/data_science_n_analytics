import numpy as np

#================================#
# Constructing Matrices #
#================================#

survey = np.array([[20, 22_000, 12], [35, 65_000, 16], [55, 19_000, 11], [45, 35_000, 12]])

print(survey)

survey.shape

np.ones((2, 5))

np.zeros((4, 2))

survey.dtype

np.ones((2, 5)).dtype

#================================#
# Math with Matrices #
#================================#

salaries = np.array([[30_000, 37_000], [42_000, 45_000], [22_000, 29_000]])
print(salaries)

# In order to convert these salaries in dollars to salaries in thousands of dollars to make the table easier to fit on a graph
salaries_in_thousands = salaries / 1000
print(salaries_in_thousands)

# Similarly, matrices can be added if they have the same size.
# If one also had a matrix of tax refunds, and wanted to calculate everyone’s total after-tax incomes, one could just add the matrices:
refunds = np.array([[5_000, 3_000], [4_000, 4_000], [8_000, 7_000]])
print(refunds)

total_income = salaries + refunds
print(total_income)
