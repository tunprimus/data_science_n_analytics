#!/usr/bin/env python3
# https://www.blog.trainindata.com/mutual-information-with-python/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os.path import realpath as realpath
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
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
FIGURE_DPI = 72
FONT_SIZE = 30
TEST_SIZE = 0.3
NUM_FEATURES = 5

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
    except Exception as exc:
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

# Compute Mutual Information
## Between 2 categorical variables
m_info_2cat = mutual_info_score(data["sex"], data["pclass"])

m_info_2cat_clf = mutual_info_classif(data["sex"].to_frame(), data["pclass"], discrete_features=[True])

## Between categorical and continuous variables
m_info_cat_cont = mutual_info_classif(data["fare"].to_frame(), data["pclass"], discrete_features=[False])

## Between 2 continuous variables
m_info_2cont = mutual_info_regression(data["fare"].to_frame(), data["age"], discrete_features=[False])

# Split Data into Training and Testing Groups
X_train, X_test, y_train, y_test = train_test_split(data.drop("survived", axis=1), data["survived"], test_size=TEST_SIZE, random_state=RANDOM_SEED)
print(X_train.columns)
## Create mask to flag categorical variables
discrete_vars = [True, True, False, True, True, False, True, True]

# Calculate the Mutual Information for the Data
mut_info = mutual_info_classif(X_train, y_train, discrete_features=discrete_vars)
print(mut_info)

# Capture Mutual Information in a Series and Graph
mut_info = pd.Series(mut_info)
mut_info.index = X_train.columns
mut_info.sort_values(ascending=False).plot.bar(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
plt.ylabel("Mutual Information", fontsize=FONT_SIZE)
plt.title("Mutual Information between Predictors and Target", fontsize=(FONT_SIZE + 10))


# Function to randomly sample from NumPy array
def random_numpy_sampler(arr, size=1):
    return arr[np.random.choice(len(arr), size=size, replace=False)]

# To Select the Best Predictor Features
best_sel = SelectKBest(mutual_info_classif, k=NUM_FEATURES).fit(X_train, y_train)
X_train_sel = best_sel.transform(X_train)
X_test_sel = best_sel.transform(X_test)
print(random_numpy_sampler(X_train_sel, RANDOM_SAMPLE_SIZE))
print(random_numpy_sampler(X_test_sel, RANDOM_SAMPLE_SIZE))

