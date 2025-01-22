#!/usr/bin/env python3
# https://www.blog.trainindata.com/mutual-information-with-python/
import pandas as pd
import numpy as np

from os.path import realpath as realpath
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# from scipy.special import agm


# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

RANDOM_SAMPLE_SIZE = 13
RANDOM_SEED = 42
GOLDEN_RATIO = 1.618033989
FIGURE_WIDTH = 30
FIGURE_HEIGHT = FIGURE_WIDTH / GOLDEN_RATIO
FIG_DPI = 72
TEST_SIZE = 0.25
NUM_ESTIMATORS = 100
MAX_DEPTH = 4
NUM_NEIGHBOURS = 10

# Load the Titanic Data
variables = ["Pclass", "Survived", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked",]

realpath_to_dataset = realpath("../000_common_dataset/Titanic-dataset-train.csv")

buffer = pd.read_csv(realpath_to_dataset, usecols=variables, na_values="?", dtype={"fare": float, "age": float })

print(buffer.head())
print(buffer.sample(RANDOM_SAMPLE_SIZE))

# Function to rename columns
def rename_columns(df):
    """
    Function to rename columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    df_copy = df.copy()
    df_copy.rename(columns=lambda x: x.lower().strip().replace(" ", "_"), inplace=True)
    return df_copy

data = rename_columns(buffer)
print(data.sample(RANDOM_SAMPLE_SIZE))

data.dropna(subset=["embarked", "fare"], inplace=True)
data["age"] = data["age"].fillna(data["age"].mean())
def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return "N"
data["cabin"] = data["cabin"].apply(get_first_cabin).str[0]

print(data.head())
print(data.sample(RANDOM_SAMPLE_SIZE))

label_encoder = LabelEncoder()
data["cabin"] = label_encoder.fit_transform(data["cabin"])
## Convert categorical variables to numbers
ord_encoder = OrdinalEncoder()
var_to_ord = ["sex", "cabin", "embarked"]
data[var_to_ord] = ord_encoder.fit_transform(data[var_to_ord])

print(data.head())
print(data.sample(RANDOM_SAMPLE_SIZE))


