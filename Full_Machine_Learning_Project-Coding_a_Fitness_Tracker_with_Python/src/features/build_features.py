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

df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]

print(duration.seconds)

unique_sets = df["set"].unique()

for s in unique_sets:
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

print(duration_df.iloc[0] / 5)
print(duration_df.iloc[1] / 10)

# --------------------------------------------------------------
# Butterworth low-pass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

# Sampling frequency
sampling_freq = 1000 / 200
cutoff_freq = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", sampling_freq, cutoff_freq, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

# Compare original data to filtered one at cutoff_freq 1.3
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="row data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# Loop over all the columns to overwrite columns with filter values
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sampling_freq, cutoff_freq, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

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

