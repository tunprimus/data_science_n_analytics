import numpy as np


# Ques_1. Using `np.arange`, create a 4 x 5 matrix with all the numbers from 0 to 19.
up_to_20_matrix = np.arange(20).reshape((4, 5))
print(up_to_20_matrix)

# Ques_2. Subset the bottom, right-most entry (the number 19) from the matrix with simple indexing.
print(up_to_20_matrix[-1, -1])

# Ques_3. Subset the entire second row of the matrix (the row starting with the number 5) with simple indexing.
print(up_to_20_matrix[1, :])

# Ques_4. Subset the third and fourth columns of the matrix (the columns starting with 2 and 3) with simple indexing.
print(up_to_20_matrix[:, 2:4])

# Ques_5. Create the matrix `survey` created above. Recall that each row of this matrix contains survey responses from a different person, where the first column contains respondent ages, the second column contains incomes, and the third column contains years of education. Subset the matrix with a logical test to get only respondents with 12 or more years of education.
survey = np.array([[20, 22_000, 12], [35, 65_000, 16], [55, 19_000, 11], [45, 35_000, 12]])
education = survey[:, 2]
education_above_12 = education >= 12
print(survey[education_above_12, :])

# Ques_6. Now, in a single line of code, subset your `survey` matrix to get the incomes of respondents with 12 or more years of education.
print(survey[survey[:, 2] >= 12, :])

# Ques_7. Now, in a single line of code, calculate the average income of respondents with 12 or more years of education.
print(np.mean(survey[survey[:, 2] >= 12, 1]))
