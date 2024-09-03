import pandas as pd
import numpy as np
from glob import glob
from os.path import expanduser as path_finder
from os.path import realpath as realpath

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

acc_df.info()
gyr_df.info()

pd.to_datetime(df["epoch (ms)"], unit="ms")
pd.to_datetime(df["time (01:00)"])

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

acc_df.info()
gyr_df.info()

print(acc_df.sample(13))
print(gyr_df.sample(13))

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

location = "~/zzz_personal/Data_Science_Analytics/Full_Machine_Learning_Project-Coding_a_Fitness_Tracker_with_Python/data/raw/MetaMotion/MetaMotion/*"
path = path_finder(location)

data_path = "Coding_a_Fitness_Tracker_with_Python/data/raw/MetaMotion/MetaMotion/"

files = glob(path)

def read_data_from_files(files):
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
    
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set"
]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last"
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

# Split by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

# Convert the set column to int16
data_resampled["set"] = data_resampled["set"].astype("int16")
data_resampled.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

real_path_to_interim = realpath("../../data/interim")

data_resampled.to_pickle(f"{real_path_to_interim}/01_data_processed.pkl")
