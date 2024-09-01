import numpy as np
from matplotlib import pyplot as plt


#================================#
# Views and Copies with Matrices #
#================================#

# Arrays are collections of data of the same type with a regular structure organised into N dimensions (which is why they are sometimes referred to as ND-Arrays instead of just arrays).

# When N=1, an array is the equivalent of a vector:

vec_arr = np.arange(20)
print(vec_arr)

# And when N=2, an array is just a matrix:

mat_arr = np.arange(20).reshape(5, 4)
print(mat_arr)


#================================#
# Three Dimensional Arrays #
#================================#

mri = np.load("data/mri_neck_vertical_slices.npy")
print(type(mri))

# This MRI is represented by a three-dimensional array. More specifically, it consists of 15 stacked images, each of which is 512 x 512 pixels
print(mri.shape)

# One can easily pull out one of these slices and visualise it
plt.figure(figsize=(8, 8))
plt.imshow(mri[7, :, :], cmap="gray", aspect="equal")

# This is the 7th slice of 15, so it is showing the approximate middle of the patient’s neck. But if one wants to move outward from the centre of the patient’s body, one could pick a slice closer to 0 or 14
plt.figure(figsize=(8, 8))
plt.imshow(mri[14, :, :], cmap="gray")

"""
Finally, because this data is three-dimensional, one also has the option of cutting the slices along a different axis. In the images above, one had specified a specific value for the first dimension of the array and plotted the second and third dimensions along the x and y-axes of the image. This has given the vertical slices of the patient’s neck.

But if, instead, one fixes a value of the second dimension, one can actually get a horizontal slice of the patient’s neck. Now, because one only has 15 slices along the first dimension of the array, the resolution along this axis is not nearly as good (a full MRI study includes taking detailed slices along all axes, and CT scans can be used to generate volumetric data with consistent resolution along all axes), but it does illustrate how working with volumetric data allows us to really explore three-dimensional objects
"""

plt.figure(figsize=(8, 8))
plt.imshow(mri[:, 300, :], cmap="gray", aspect=10)
