from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np


#================================#
# Matrices as Images #
#================================#

# Load image as pixel array
jellyfish = image.imread("../img/jellyfish.png")
plt.imshow(jellyfish, cmap="gray")

print(type(jellyfish))

jellyfish.shape

print(jellyfish)


#================================#
# Image Manipulation #
#================================#

darker = jellyfish * 0.75
plt.imshow(darker, cmap="gray", vmin=0, vmax=1)

# Convert Greyscale to Two-Tone
two_tone = jellyfish > 0.3
plt.imshow(two_tone, cmap="gray")

#================================#
# Adjusting Contrast #
#================================#

# Square root function
zero_to_one = np.arange(0, 1, 0.01)
plt.plot(zero_to_one, np.sqrt(zero_to_one))

# Function to enhance contrast
def change_contrast(image):
    # Shift values so they lie between -1 and 1
    re_centre_and_re_scale = (image - 0.5) * 2    
    
    # Take square root of absolute value
    # (the square root of the negative number is undefined)
    squared_values = np.abs(re_centre_and_re_scale) ** 0.5
    
    # Make the negative values negative again
    correctly_signed_and_squared = squared_values * np.sign(re_centre_and_re_scale)

    # Shift back and rescale
    shifted_and_scaled_back = (correctly_signed_and_squared / 2) + 0.5
    return shifted_and_scaled_back

plt.imshow(change_contrast(jellyfish), cmap="gray", vmin=0, vmax=1)

# Compared to our original image:
plt.imshow(jellyfish, cmap="gray")


#================================#
# Generalising Contrast Adjustment #
#================================#

# Function to variably enhance contrast
def change_contrast(image, exponent=0.5):
    # Shift values so they lie between -1 and 1
    re_centre_and_re_scale = (image - 0.5) * 2    
    
    # Take square root of absolute value
    # (the square root of the negative number is undefined)
    squared_values = np.abs(re_centre_and_re_scale) ** exponent
    
    # Make the negative values negative again
    correctly_signed_and_squared = squared_values * np.sign(re_centre_and_re_scale)

    # Shift back and rescale
    shifted_and_scaled_back = (correctly_signed_and_squared / 2) + 0.5
    return shifted_and_scaled_back

plt.imshow(change_contrast(jellyfish, 0.25), cmap="gray", vmin=0, vmax=1)

# Decrease contrast with an exponent greater than one
plt.imshow(change_contrast(jellyfish, exponent=2), cmap="gray", vmin=0, vmax=1)
