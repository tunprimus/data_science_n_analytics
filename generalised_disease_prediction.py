#!/usr/bin/env python3
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from os.path import realpath as realpath


np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

RANDOM_SAMPLE_SIZE = 13
RANDOM_SEED = 42
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 30
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_DPI = 72
FONT_SIZE = 30
TEST_SIZE = 0.23
NUM_ESTIMATORS = 100
MAX_DEPTH = 4
NUM_NEIGHBOURS = 10


# Function to get training data
def get_training_data(data_path_train):
    real_path_data_train = realpath(data_path_train)
    train_data = pd.read_csv(real_path_data_train).dropna(axis = 1)
    return train_data

# Function to get test data
def get_testing_data(data_path_test):
    real_path_data_test = realpath(data_path_test)
    test_data = pd.read_csv(real_path_data_test).dropna(axis = 1)
    return test_data

# Function to check and display if dataset is balanced around target category
def check_dataset_balance(data_to_check, target_category="prognosis"):
    disease_counts = data_to_check[target_category].value_counts()
    temp_df = pd.DataFrame({
        "Disease": disease_counts.index,
        "Counts": disease_counts.values
    })
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    sns.barplot(x = "Disease", y = "Counts", data = temp_df)
    plt.xticks(rotation = 90)
    plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    plt.xlabel(f"{temp_df.columns[0]}", fontsize=FONT_SIZE)
    plt.ylabel(f"{temp_df.columns[1]}", fontsize=FONT_SIZE)
    plt.title(f"Plot of Dataset Balance Around {target_category.title()}", fontsize=(FONT_SIZE + 10),)
    plt.show()


# Function to encode target category into numerical with LabelEncoder
def encode_target_category(data_to_use, target_category):
    label_encoder = LabelEncoder()
    data_to_use[target_category] = label_encoder.encode(data_to_use[target_category])


# Function to split the data for training and testing the model
def split_data(data_to_use, target_category):
    X = data_to_use.drop(target_category)
    y = data_to_use[target_category]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state = RANDOM_SEED)
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
    return X_train, X_test, y_train, y_test

# Define scoring metric for k-fold cross validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Function using SVM Classifier to train and test
def svm_classifier(data_to_use, target_category):
    X_train, X_test, y_train, y_test = split_data(data_to_use, target_category)
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    preds = svm_model.predict(X_test)
    acc_score_train = accuracy_score(y_train, svm_model.predict(X_train)) * 100
    acc_score_test = accuracy_score(y_test, preds) * 100
    print(f"Accuracy on train data by SVM Classifier: {acc_score_train}")
    print(f"Accuracy on test data by SVM Classifier: {acc_score_test}")
    cf_matrix = confusion_matrix(y_test, preds)
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    sns.heatmap(cf_matrix, annot=True)
    plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    plt.xlabel(f"{str(preds).title()}", fontsize=FONT_SIZE)
    plt.ylabel(f"{str(y_test).title()}", fontsize=FONT_SIZE)
    plt.title("Confusion Matrix for SVM Classifier on Test Data", fontsize=(FONT_SIZE + 10),)
    plt.show()


# Function using Naive Bayes Classifier to train and test
def nb_classifier(data_to_use, target_category):
    X_train, X_test, y_train, y_test = split_data(data_to_use, target_category)
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    preds = nb_model.predict(X_test)
    acc_score_train = accuracy_score(y_train, nb_model.predict(X_train)) * 100
    acc_score_test = accuracy_score(y_test, preds) * 100
    print(f"Accuracy on train data by Naive Bayes Classifier: {acc_score_train}")
    print(f"Accuracy on test data by Naive Bayes Classifier: {acc_score_test}")
    cf_matrix = confusion_matrix(y_test, preds)
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    sns.heatmap(cf_matrix, annot=True)
    plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    plt.xlabel(f"{str(preds).title()}", fontsize=FONT_SIZE)
    plt.ylabel(f"{str(y_test).title()}", fontsize=FONT_SIZE)
    plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data", fontsize=(FONT_SIZE + 10),)
    plt.show()


# Function using Random Forest Classifier to train and test
def rf_classifier(data_to_use, target_category):
    X_train, X_test, y_train, y_test = split_data(data_to_use, target_category)
    rf_model = RandomForestClassifier(random_state=RANDOM_SEED)
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)
    acc_score_train = accuracy_score(y_train, rf_model.predict(X_train)) * 100
    acc_score_test = accuracy_score(y_test, preds) * 100
    print(f"Accuracy on train data by Random Forest Classifier: {acc_score_train}")
    print(f"Accuracy on test data by Random Forest Classifier: {acc_score_test}")
    cf_matrix = confusion_matrix(y_test, preds)
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    sns.heatmap(cf_matrix, annot=True)
    plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
    plt.xlabel(f"{str(preds).title()}", fontsize=FONT_SIZE)
    plt.ylabel(f"{str(y_test).title()}", fontsize=FONT_SIZE)
    plt.title("Confusion Matrix for Random Forest Classifier on Test Data", fontsize=(FONT_SIZE + 10),)
    plt.show()


# Function that combines models for training on data
def combine_models_trainer(data_to_use, target_category, models_to_use=[]):
    pass











def main():
    pass

if __name__ == "__main__":
    main()
