import numpy as np


# Ques_1. Using `np.arange`, create a 3 x 5 matrix with all the numbers from 0 to 14. Assignment to the variable `my_matrix`.
my_matrix = np.arange(15).reshape((3, 5))
print(my_matrix)

# Ques_2. Subset the third and fourth columns of the matrix (the columns starting with 2 and 3) with simple indexing. Assign the subset to the variable `m2`.
m2 = my_matrix[:, 2:4]
print(m2)

# Ques_3. Change the top, left-most element of your new matrix `m2` to -99.
m2[0, 0] = -99
print(m2)

# Ques_4. Without running any code, try and predict what `my_matrix` currently looks like.
print("my_matrix is now [0, 1, -99, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]")

# Ques_5. Now check `my_matrix`—does it look how you expected? Why or why not?
print(my_matrix)
