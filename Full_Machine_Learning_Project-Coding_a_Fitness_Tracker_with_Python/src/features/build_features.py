import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
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

subset01 = df_lowpass[df_lowpass["set"] == 45]
print(subset01["label"][0])

# Compare original data to filtered one at cutoff_freq 1.3
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset01["acc_y"].reset_index(drop=True), label="row data")
ax[1].plot(subset01["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
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

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# Elbow technique to determine optimal number of PCA components
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset02 = df_pca[df_pca["set"] == 35]
print(subset02["label"][0])
subset02[["pca_1", "pca_2", "pca_3"]].plot()

subset03 = df_pca[df_pca["set"] == 23]
print(subset03["label"][0])
subset03[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = (df_squared["acc_x"] ** 2) + (df_squared["acc_y"] ** 2) + (df_squared["acc_z"] ** 2)
gyr_r = (df_squared["gyr_x"] ** 2) + (df_squared["gyr_y"] ** 2) + (df_squared["gyr_z"] ** 2)

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset04 = df_squared[df_squared["set"] == 14]
subset04[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()

updated_predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws01 = int(1000 / 200)

for col in updated_predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws01, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws01, "std")
print(df_temporal)
print(df_temporal.sample(5))
print(df_temporal.head(7))

# Prevent introducing data from a different exercise type
df_temporal_list = []
unique_temporal_set = df_temporal["set"].unique()

for s in unique_temporal_set:
    print(f"Applying temporal abstraction to set {s}")
    subset05 = df_temporal[df_temporal["set"] == s].copy()
    for col in updated_predictor_columns:
        subset05 = NumAbs.abstract_numerical(subset05, [col], ws01, "mean")
        subset05 = NumAbs.abstract_numerical(subset05, [col], ws01, "std")
    df_temporal_list.append(subset05)

df_temporal = pd.concat(df_temporal_list)
df_temporal.info()

subset06 = df_temporal[df_temporal["set"] == 13]
subset06[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset06[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot(subplots=True)
subset06[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
freqAbs = FourierTransformation()

sampling_freq = int(1000 / 200)
ws02 = int(2800 / 200)

df_freq = freqAbs.abstract_frequency(df_freq, ["acc_y"], ws02, sampling_freq)
print(df_freq.columns)

subset07 = df_freq[df_freq["set"] == 15]
subset07[["acc_y"]].plot()
subset07[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14"
    ]
    ].plot()

# Run loop for each column
df_freq_list = []
unique_freq_set = df_freq["set"].unique()

for s in unique_freq_set:
    print(f"Applying Fourier transformation to set {s}")
    subset08 = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset08 = freqAbs.abstract_frequency(subset08, updated_predictor_columns, ws02, sampling_freq)
    df_freq_list.append(subset08)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
print(df_freq.columns)



# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()

df_freq_halved = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

