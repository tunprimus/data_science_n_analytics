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


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------