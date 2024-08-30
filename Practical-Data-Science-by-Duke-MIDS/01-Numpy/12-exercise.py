import numpy as np


# Ques_1. Using `np.arange`, create a vector with all the integers from 0 to 23.
arr_vec = np.arange(24)
print(arr_vec)

# Ques_2. Using `.reshape`, convert that vector into a matrix with 4 rows and 6 columns where the numbers increase as you move from left to right along each row before wrapping to the next row.
arr_vec_reshaped = arr_vec.reshape((4, 6))
print(arr_vec_reshaped)

# Ques_3. Using `reshape`, try to convert this matrix into a 5 x 5 matrix. Why were you unsuccessful?
# arr_vec_wrong_reshape = arr_vec.reshape((5, 5))
# print(arr_vec_wrong_reshape)

# Ques_4. Using `np.arange`, create a new sequence that you can reshape into a 5 x 5 matrix.
arr_vec_5_by_5 = np.arange(25).reshape((5, 5))
print(arr_vec_5_by_5)
