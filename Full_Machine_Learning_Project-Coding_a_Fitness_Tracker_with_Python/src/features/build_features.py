import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from os.path import realpath as realpath
# Monkey patching NumPy for compatibility with newer versions
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)



# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

real_path_to_pickle02 = realpath("../../data/interim/02_outliers_removed_chauvenet.pkl")

df = pd.read_pickle(real_path_to_pickle02)

df_columns = df.columns
predictor_columns = list(df_columns[0:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 72
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()

for col in predictor_columns:
    df[col] = df[col].interpolate()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------


# --------------------------------------------------------------
# Butterworth low-pass filter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

