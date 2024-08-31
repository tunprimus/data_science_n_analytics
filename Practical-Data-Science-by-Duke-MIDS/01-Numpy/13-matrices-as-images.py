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
