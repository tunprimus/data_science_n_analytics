import pandas as pd
import numpy as np
from glob import glob
from os.path import expanduser as path_finder

pd.set_option("mode.copy_on_write", True)

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv("~/zzz_personal/Data_Science_Analytics/Full_Machine_Learning_Project-Coding_a_Fitness_Tracker_with_Python/data/raw/MetaMotion/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
print(single_file_acc)

single_file_gyr = pd.read_csv("~/zzz_personal/Data_Science_Analytics/Full_Machine_Learning_Project-Coding_a_Fitness_Tracker_with_Python/data/raw/MetaMotion/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")
print(single_file_gyr)


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
location = "~/zzz_personal/Data_Science_Analytics/Full_Machine_Learning_Project-Coding_a_Fitness_Tracker_with_Python/data/raw/MetaMotion/MetaMotion/*"
path = path_finder(location)

files = glob(path)
print(len(files))

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = "Coding_a_Fitness_Tracker_with_Python/data/raw/MetaMotion/MetaMotion/"
print(data_path)

f = files[0]
print(f)
buffer_arr = f.split("-")
print(buffer_arr)

participant = buffer_arr[1].replace(data_path, "")
label = buffer_arr[2]
category = buffer_arr[3].split("_")[0].rstrip("123456789")

print(participant, label, category)

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    buffer_arr = f.split("-")

    participant = buffer_arr[1].replace(data_path, "")
    label = buffer_arr[2]
    category = buffer_arr[3].split("_")[0].rstrip("123456789")

    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
    elif "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])
    else:
        continue


print(acc_df)
print(acc_df.sample(13))
print(gyr_df)
print(gyr_df.sample(13))

print(acc_df[acc_df["set"] == 7])
print(gyr_df[gyr_df["set"] == 13])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
