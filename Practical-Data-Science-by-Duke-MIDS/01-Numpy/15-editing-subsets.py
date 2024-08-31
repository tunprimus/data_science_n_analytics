import numpy as np


#================================#
# Editing Subsets #
#================================#

survey = np.array([[20, 22_000, 12], [35, 65_000, 16], [55, 19_000, 11], [45, 35_000, 12]])
print(survey)

# This modifies all entries
print(survey * 1.02)

# Instead, extract the column with income, modify it, then replace the old income column with the updated column:

# Extract income
income_column = survey[:, 1]
# Adjust income
adjusted_income = income_column * 1.02
# Replace income with new values!
survey[:, 1] = adjusted_income
print(survey)

# Or, if we wanted, we could actually do all this in one step:

# Re-make survey so it has not been adjusted for inflation
survey = np.array([[20, 22_000, 12], [35, 65_000, 16], [55, 19_000, 11], [45, 35_000, 12]])

# Now adjust income in one step!
survey[:, 1] = survey[:, 1] * 1.02
print(survey)

# If one wants to see what people’s incomes would look like if anyone who did not finish high school (education < 12) got a tax credit of 10,000 dollars:
survey[survey[:, 2] < 12, 1] = survey[survey[:, 2] < 12, 1] + 10000
print(survey)

#================================#
# Views and Copies with Matrices #
#================================#

# When it comes to views and copies, the same rules apply to matrices as applied to vectors: if one creates a subset through simple indexing, the result will be a view; if one creates a subset by a different method, one gets a copy!
