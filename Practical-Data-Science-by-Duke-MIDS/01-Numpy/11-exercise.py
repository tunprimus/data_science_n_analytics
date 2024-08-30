import numpy as np


# Ques_1. Using `np.array`, create a new matrix with two rows and three columns. In the first row, place information about yourself, and in the second row place information about a good friend. In the first column enter your ages, in the second column enter an estimate of your current income rounded to the nearest thousand, and in the third column add a `1` if you identify as a woman and a `0` otherwise.
people_array = np.array([
    [25, 100_000, 0],
    [30, 80_000, 1]
])

# Ques_2. Confirm the shape of your matrix with `.shape`.
print(people_array.shape)

# Ques_3. What data type is your matrix?
print(people_array.dtype)

# Ques_4. Without writing any code, what data type matrix do you think you would get if, instead of rounding your income to the nearest thousand, you entered your income to the nearest cent (or your local currency’s decimal equivalent—Indian paise, British pence, etc.)?
people_array2 = np.array([
    [25, 100_000.50, 0],
    [30, 80_000.70, 1]
])

# Ques_5. Check your answer to number 4 above in Python.
print(people_array2.dtype)
