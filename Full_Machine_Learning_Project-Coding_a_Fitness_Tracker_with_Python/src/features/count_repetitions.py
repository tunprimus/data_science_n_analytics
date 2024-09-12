import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error
from os.path import realpath as realpath

# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)
pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 72
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

real_path_to_pickle01 = realpath("../../data/interim/01_data_processed.pkl")

df = pd.read_pickle(real_path_to_pickle01)

# Remove rest data points from the DataFrame
df = df[df["label"] != "rest"]

# Add scalar solution of the vectors
acc_r = (df["acc_x"] ** 2) + (df["acc_y"] ** 2) + (df["acc_z"] ** 2)
gyr_r = (df["gyr_x"] ** 2) + (df["gyr_y"] ** 2) + (df["gyr_z"] ** 2)

df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

# Create separate DataFrames for each exercise type
bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]

# --------------------------------------------------------------
# Visualise data to identify patterns
# --------------------------------------------------------------

plot_df = dead_df
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()

# ====================================================== #
# bench_df -> acc_y or gyr_r
# squat_df -> acc_r or gyr_r
# row_df -> acc_x, acc_y, acc_z or gyr_x
# ohp_df -> acc_x or gyr_z
# dead_df -> acc_x, acc_r, gyr_z or gyr_r
# ====================================================== #

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

# Sampling frequency
sampling_freq = 1000 / 200
cutoff_freq = 0.4

LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

bench_set["acc_r"].plot()

column = "acc_r"
# low_pass_filter(
#         self,
#         data_table,
#         col,
#         sampling_frequency,
#         cutoff_frequency,
#         order=5,
#         phase_shift=True,
#     )
LowPass.low_pass_filter(bench_set, column, sampling_freq, cutoff_freq, order=5)[column + "_lowpass"].plot()
LowPass.low_pass_filter(squat_set, column, sampling_freq, cutoff_freq, order=5)[column + "_lowpass"].plot()
LowPass.low_pass_filter(row_set, column, sampling_freq, cutoff_freq, order=5)[column + "_lowpass"].plot()
LowPass.low_pass_filter(ohp_set, column, sampling_freq, cutoff_freq, order=5)[column + "_lowpass"].plot()
LowPass.low_pass_filter(dead_set, column, sampling_freq, cutoff_freq, order=5)[column + "_lowpass"].plot()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

df_columns = df.columns
predictor_columns = list(df_columns[0:6])
predictor_columns.extend(list(df_columns[-2:]))
print(predictor_columns)

# Loop over all the columns to to add filter values without overwrite
for col in predictor_columns:
    df = LowPass.low_pass_filter(df, col, sampling_freq, cutoff_freq, order=5)
print(df)
print(df.columns)

def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    data = LowPass.low_pass_filter(data_table=dataset, col=column, sampling_frequency=sampling_freq, cutoff_frequency=cutoff, order=order)

    indices = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indices]

    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()

    return len(peaks)

def count_reps_extended(dataset, cutoff=0.4, order=10, column=predictor_columns[-2]):
    if (len(column) == 1):
        data = LowPass.low_pass_filter(data_table=dataset, col=column, sampling_frequency=sampling_freq, cutoff_frequency=cutoff, order=order)

        indices = argrelextrema(data[column + "_lowpass"].values, np.greater)
        peaks = data.iloc[indices]

        fig, ax = plt.subplots()
        plt.plot(dataset[f"{column}_lowpass"])
        plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
        ax.set_ylabel(f"{column}_lowpass")
        exercise = dataset["label"].iloc[0].title()
        category = dataset["category"].iloc[0].title()
        plt.title(f"{category} {exercise}: {len(peaks)} Reps")
        plt.show()

        return len(peaks)
    else:
        peaks_buffer = []
        fig_01 = plt.figure(figsize=(30, 20))
        fig_02 = plt.figure(figsize=(30, 20))
        for i, col in enumerate(predictor_columns):
            data = LowPass.low_pass_filter(data_table=dataset, col=col, sampling_frequency=sampling_freq, cutoff_frequency=cutoff, order=order)

            indices = argrelextrema(data[col].values, np.greater)
            indices_lowpass = argrelextrema(data[col + "_lowpass"].values, np.greater)
            peaks = data.iloc[indices]
            peaks_lowpass = data.iloc[indices_lowpass]
            
            data_columns00 = data.columns
            data_columns01 = list(data_columns00[:6])
            data_columns02 = list(data_columns00[10:])
            data_columns = data_columns01 + data_columns02
            
            data_column_len = len(data_columns)
            side_length = (np.ceil(data_column_len**(1/2))).astype("int16")
            gs = gridspec.GridSpec(side_length, side_length)

            ax_01 = fig_01.add_subplot(gs[i])
            ax_02 = fig_02.add_subplot(gs[i])
            ax_01.plot(dataset[f"{col}"])
            ax_01.plot(peaks[f"{col}"], "o", color="red")
            ax_01.set_ylabel(f"{col}")
            
            ax_02.plot(dataset[f"{col}_lowpass"])
            ax_02.plot(peaks_lowpass[f"{col}_lowpass"], "o", color="red")
            ax_02.set_ylabel(f"{col}_lowpass")
            exercise = dataset["label"].iloc[0].title()
            category = dataset["category"].iloc[0].title()
            plt.title(f"{category} {exercise}: {len(peaks_lowpass)} Reps")
            peaks_buffer.append(len(peaks_lowpass))
    return max(peaks_buffer)

# Cut-off frequency only accurate for bench press and dead-lift
count_reps(bench_set)
count_reps(squat_set)
count_reps(row_set)
count_reps(ohp_set)
count_reps(dead_set)

# Fine tuning the cut-off frequency
count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(row_set, cutoff=0.65, column="gyr_x")
count_reps(ohp_set, cutoff=0.35)
count_reps(dead_set, cutoff=0.4)

# Further fine-tuning

# ====================================================== #
# bench_df -> acc_y or gyr_r
# squat_df -> acc_r or gyr_r
# row_df -> acc_x, acc_y, acc_z or gyr_x
# ohp_df -> acc_x or gyr_z
# dead_df -> acc_x, acc_r, gyr_z or gyr_r
# ====================================================== #


count_reps_extended(bench_set, column="acc_y")
count_reps_extended(squat_set, cutoff=0.35, column="acc_r")
count_reps_extended(row_set, cutoff=0.65, column="acc_x")
count_reps_extended(ohp_set, cutoff=0.35, column="acc_x")
count_reps_extended(dead_set, cutoff=0.4, column="acc_r")

# --------------------------------------------------------------
# Create benchmark DataFrame
# --------------------------------------------------------------

# Create a repetition column
df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)

# Create a categorised benchmark DataFrame
rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()
rep_df["reps_pred"] = 0

unique_set = df["set"].unique()

# Loop through the set in original DataFrame to predict repetitions
for s in unique_set:
    subset_01 = df[df["set"] == s]
    column = "acc_r"
    cutoff=0.4
    # Change cutoff value and column to enhance accuracy
    if subset_01["label"].iloc[0] == "bench":
        column = "acc_y"

    if subset_01["label"].iloc[0] == "squat":
        cutoff = 0.35

    if subset_01["label"].iloc[0] == "row":
        cutoff = 0.65
        column = "acc_x"

    if subset_01["label"].iloc[0] == "ohp":
        cutoff = 0.35
        column = "gyr_z"
        
    if subset_01["label"].iloc[0] == "dead":
        column = "gyr_z"
    # Count the number of repetitions
    reps = count_reps(subset_01, cutoff=cutoff, column=column)
    # Add predicted repetitions to `reps_pred` column
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

print(rep_df)

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)
rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().plot.bar()
