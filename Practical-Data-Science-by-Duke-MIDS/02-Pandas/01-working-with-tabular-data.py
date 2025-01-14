#!/usr/bin/env python3
import numpy as np
import pandas as pd


#================================#
# Working with tabular data #
#================================#

a = np.array([6.1,150.0,25,'A-'])
b = np.array([5.6,122.0,29,'B+'])
c = a + b

# Pandas has no problem with mixed data types
a = pd.Series([6.1, 150.0, 25, "A-"])
b = pd.Series([5.6, 122.0, 29, "B+"])
c = a + b
print(c)
