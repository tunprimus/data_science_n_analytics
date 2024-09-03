import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import realpath as realpath
from IPython.display import display

pd.set_option("mode.copy_on_write", True)

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

real_path_to_pickle = realpath("../../data/interim/01_data_processed.pkl")

df = pd.read_pickle(real_path_to_pickle)

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

# Get a subset of the data
set_df = df[df["set"] == 1]

# Plot a sample column
plt.plot(set_df["acc_y"])
# Reset index from timestamp
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

# Get all the types of exercises with .unique()
exercise_labels = df["label"].unique()

# Plot acc_y for each type of exercise
for label in exercise_labels:
    subset = df[df["label"] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# Plot 100 sample of acc_y for each type of exercise
for label in exercise_labels:
    subset = df[df["label"] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"
all_axes_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig, ax = plt.subplots()
all_axes_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

exercise_labels = df["label"].unique()
participant_labels = df["participant"].unique()

# For accelerometer data
for label in exercise_labels:
    for participant in participant_labels:
        all_axes_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
            )
        
        if (len(all_axes_df) > 0):
            fig, ax = plt.subplots()
            all_axes_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_xlabel("samples")
            ax.set_ylabel("acc")
            plt.title(f"{label} {participant}".title())
            plt.legend()


# For gyroscope data
for label in exercise_labels:
    for participant in participant_labels:
        all_axes_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
            )
        
        if (len(all_axes_df) > 0):
            fig, ax = plt.subplots()
            all_axes_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_xlabel("samples")
            ax.set_ylabel("gyr")
            plt.title(f"{label} {participant}".title())
            plt.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------