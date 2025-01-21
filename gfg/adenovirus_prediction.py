#!/usr/bin/env python3
# Importing Libraries and Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from os.path import realpath as realpath
from sklearn import tree
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)
pd.set_option("display.max_columns", None)

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

realpath_to_dataset = realpath("../000_common_dataset/adenoviruses_dataset.csv")

buffer = pd.read_csv(realpath_to_dataset)

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

adenoviruses = rename_columns(buffer)
print(adenoviruses.sample(RANDOM_SAMPLE_SIZE))

# Check the Data Info
print(adenoviruses.info())

# Check the Descriptive Statistical View of the Data
print(adenoviruses.describe(include="all"))

# Feature/Data Transformation
encoder = LabelEncoder()

def label_encoder(df, enc_type=None):
    for col in df.columns:
        if not enc_type:
            encoder = LabelEncoder()
        else:
            encoder = enc_type
        df[col] = encoder.fit_transform(df[col])
    return df

adenoviruses = label_encoder(adenoviruses, encoder)
print(adenoviruses.sample(RANDOM_SAMPLE_SIZE))

# Correlation between Feature
corr = adenoviruses.corr()
corr.style.background_gradient(cmap="coolwarm", axis=None)

# Split the Dataset
X = adenoviruses.drop("adenoviruses", axis=1)
y = adenoviruses["adenoviruses"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Build the Models

## A. Logistic Regression
lr_model = LogisticRegression()
### fit the model
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
### accuracy / score
acc_logreg = lr_model.score(X_test, y_test) * 100
print(f"Accuracy Score: {acc_logreg:.2f}%")

## B. Random Forest
rf_model = RandomForestRegressor()
### fit the model
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
### accuracy / score
acc_rf = rf_model.score(X_test, y_test) * 100
print(f"Accuracy Score: {acc_rf:.2f}%")

## C. Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=NUM_ESTIMATORS, max_depth=MAX_DEPTH)
### fit the model
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)
### accuracy / score
acc_gb = gb_model.score(X_test, y_test) * 100
print(f"Accuracy Score: {acc_gb:.2f}%")

## D. KNeighbours Classifier
knn_model = KNeighborsClassifier(n_neighbors=NUM_NEIGHBOURS)
### fit the model
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
### accuracy / score
acc_knn = knn_model.score(X_test, y_test) * 100
print(f"Accuracy Score: {acc_knn:.2f}%")

## E. Decision Tree Classifier
dt_model = tree.DecisionTreeClassifier()
### fit the model
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
### accuracy / score
acc_dt = dt_model.score(X_test, y_test) * 100
print(f"Accuracy Score: {acc_dt:.2f}%")

## F. Gaussian Neighbors(GaussianNB)
gnb_model = GaussianNB()
### fit the model
gnb_model.fit(X_train, y_train)
y_pred = gnb_model.predict(X_test)
### accuracy / score
acc_gnb = gnb_model.score(X_test, y_test) * 100
print(f"Accuracy Score: {acc_gnb:.2f}%")

## G. Support Vector Machines(SVM)
svm_model = svm.SVC(kernel="linear")
### fit the model
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
### accuracy / score
acc_svm = svm_model.score(X_test, y_test) * 100
print(f"Accuracy Score: {acc_svm:.2f}%")

# Compile Model Accuracy
models = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting Regressor", "KNeighbors Classifier", "Decision Tree Classifier", "Gaussian Neighbors", "Support Vector Machines"],
    "Score": [acc_logreg, acc_rf, acc_gb, acc_knn, acc_dt, acc_gnb, acc_svm]
})
print(models.sort_values(by="Score", ascending=False))

