from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np


#================================#
# Matrices as Images #
#================================#

# Load image as pixel array
jellyfish = image.imread("../img/jellyfish.png")
plt.imshow(jellyfish, cmap="gray")

type(jellyfish)
