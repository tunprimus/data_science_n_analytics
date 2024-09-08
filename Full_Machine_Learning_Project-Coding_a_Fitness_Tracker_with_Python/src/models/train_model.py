import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from os.path import realpath as realpath
# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)



# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
# plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.dpi"] = 72
plt.rcParams["lines.linewidth"] = 2


# Load the data
real_path_to_pickle03 = realpath("../../data/interim/03_data_features.pkl")

df = pd.read_pickle(real_path_to_pickle03)

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set"], axis=1)

X = df_train.drop(["label"], axis=1)
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(kind="bar", ax=ax, color="lightblue", label="Total")
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
df_train_columns = df_train.columns
time_features = [t for t in df_train_columns if "_temp_" in t]
freq_features = [f for f in df_train_columns if ("_freq" in f) or ("_pse" in f)]
cluster_features = ["cluster"]

print(f"Basic features: {len(basic_features)}")
print(f"Squared features: {len(square_features)}")
print(f"PCA features: {len(pca_features)}")
print(f"Time features: {len(time_features)}")
print(f"Frequency features: {len(freq_features)}")
print(f"Cluster features: {len(cluster_features)}")

feature_set_01 = list(set(basic_features))
feature_set_02 = list(set(feature_set_01 + square_features + pca_features))
feature_set_03 = list(set(feature_set_02 + time_features))
feature_set_04 = list(set(feature_set_03 + freq_features + cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
