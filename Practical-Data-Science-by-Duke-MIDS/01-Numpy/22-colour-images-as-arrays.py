from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np


#================================#
# Colour Images as Arrays #
#================================#

# Load image as pixel array
jellyfish = image.imread("../img/jellyfish_colour.png")
plt.imshow(jellyfish)

print(jellyfish.shape)

print(np.min(jellyfish), np.max(jellyfish))

#================================#
# Breaking Down Our Image #
#================================#
# Picking the blue layer
plt.imshow(jellyfish[:, :, 2], cmap="gray")

# Picking the red layer
plt.imshow(jellyfish[:, :, 0], cmap="gray")


#================================#
# A true grayscale image #
#================================#

# Greyscale image is mean of intensities of RGB channels
# For this image, take the mean across the 3rd axis
grey_jellyfish = np.mean(jellyfish, axis=2)
print(grey_jellyfish.shape)

plt.imshow(grey_jellyfish,cmap="gray")


#================================#
# Subsetting image matrices #
#================================#

# For the greyscale image
small_jellyfish_grey = grey_jellyfish[500:1200,600:1300]
plt.imshow(small_jellyfish_grey,cmap="gray")

# For the colour image
small_jellyfish = jellyfish[500:1200,600:1300,:]
plt.imshow(small_jellyfish)

