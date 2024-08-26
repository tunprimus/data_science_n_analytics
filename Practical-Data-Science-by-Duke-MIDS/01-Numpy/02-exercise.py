import numpy as np


# Ques_1. Suppose the following were the heart rates reported by your Fitbit over the day: 68, 65, 77, 110, 160, 161, 162, 161, 160, 161, 162, 163, 164, 163, 162, 100, 90, 97, 72, 60, 70. Put these numbers into a numpy array.
heart_rates_array = np.array([68, 65, 77, 110, 160, 161, 162, 161, 160, 161, 162, 163, 164, 163, 162, 100, 90, 97, 72, 60, 70])
print(heart_rates_array)

# Ques_2. A commonly used measure of health is a person’s resting heart rate (basically, how low your heart rate goes when you aren’t doing anything). Find the minimum heart rate you experienced over the day.
print(np.min(heart_rates_array))

# Ques_3. One measure of exercise intensity is your maximum heart rate—suppose that during the day these data were collected, you are deliberately exercising. Find your maximum heart rate.
print(np.max(heart_rates_array))

# Ques_4. Let’s try to calculate the share of readings that were taken when you were exercising. First, create a new vector that takes on the value of True when your heart rate is above 120, and False when your heart rate is below 120.
above_120 = heart_rates_array > 120
print(above_120)
below_120 = heart_rates_array < 120
print(below_120)

# Ques_5. Now use a summarising function to calculate the share of observations for which your heart rate was above 120!
print(np.sum(above_120))
