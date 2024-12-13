import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import watermark as watermark
from os.path import realpath as realpath

# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

# pd.set_option("mode.copy_on_write", True)

RANDOM_SAMPLE_SIZE = 13
GOLDEN_RATIO = 1.618
FIGURE_WIDTH = 30
FIGURE_HEIGHT = FIGURE_WIDTH / GOLDEN_RATIO

realpath_to_dataset = realpath("./account_transfer_dataset.csv")

## Data Dictionary
## id: Unique identifier for each entry.
## entry_date: Date the accounting entry is made.
## debit_account: Accounting account to be debited.
## credit_account: Accounting account to be credited.
## amount: Monetary value of the entry.
## document: Supporting documentation for the transaction.
## transaction_nature: Description of the accounting event.
## cost_center: Department responsible for the transaction.
## taxes: Taxes and duties involved, if applicable.
## currency: Currency used in the transaction, if applicable.
## exchange_rate: Conversion rate to the national currency, if applicable.


## Load the dataset
df = pd.read_csv(realpath_to_dataset)
df.rename(columns={"cost_center": "cost_centre"}, inplace=True)
print(df.shape)
print(df.head(5))
print(df.sample(RANDOM_SAMPLE_SIZE))
print(df.columns)


## Exploratory Analysis Before Data Cleaning

print(df.info())
print(df.describe())
print(df.describe(include=[np.number]))
print(df.describe(include=[np.object]))
print(df.describe(include=[np.object]).transpose())

### Any missing value?
df.isna().any()
print(df.isna().any())

### Any missing value? How many?
df.isna().sum()
print(df.isna().sum())

### Proportion of missing value?
missing_values = df.isna().sum()
total_rows = len(df)
missing_value_proportion = missing_values / total_rows
print(missing_value_proportion)
missing_value_percentage = missing_values * 100 / total_rows
print(missing_value_percentage.round(1))

#### Generally, for up to 30% missing data, it's best to directly address the gaps and each should be treated individually.
#### If missing values exceed 50%, it may be advisable to discard the variable
#### When encountering special characters or irregular entries like ?, they won't show up in the NaN count, making them harder to detect. While NaN values are simple to address through counting and imputation using built-in pandas functions, unusual characters add complexity.


### Plot the distribution of transaction values
plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
sns.histplot(df["amount"], kde=True, bins=30)
plt.title("Distribution of Transaction Values")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

### Analysing Transaction Values Over Time
plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
df["release_date"] = pd.to_datetime(df["release_date"])
sns.lineplot(x="release_date", y="amount", data=df)
plt.title("Transaction Values Over Time")
plt.xlabel("Release Date")
plt.ylabel("Amount")
plt.xticks(rotation=45)
plt.show()

### Tax boxplot and violin plot
plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
sns.boxplot(x=df["taxes"])
sns.violinplot(x=df["taxes"])
plt.show()

### Currency countplot
plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
sns.countplot(x="currency", data=df)
plt.title("Count of Transactions by Currency")
plt.xlabel("Currency")
plt.ylabel("Count")
plt.show()

## Handling Missing Values for Numerical Variables
column_names = df.columns
for column in column_names:
    missing_val = df[f"{column}"].isna().sum()
    if missing_val > 0:
        print(f"{column} has {missing_val} missing values")
        print(f"`{column}` missing value proportion: {(missing_val / len(df)).round(3)}")
        print(f"`{column}` missing value percentage: {(missing_val * 100 / len(df)).round(1)}")

### Plot the distribution of tax values
plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
sns.histplot(df["taxes"], kde=True, bins=30)
plt.title("Distribution of Tax Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print(df["taxes"].mean())
print(df["taxes"].median())

### Replace missing values in taxes with median
# df["taxes"].fillna(df["taxes"].median(), inplace=True)
df["taxes"] = np.where(df["taxes"].isnull(), df["taxes"].median(), df["taxes"])

df["taxes"].isna().sum()

### Plot the distribution of tax values post missing value imputation
plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
sns.histplot(df["taxes"], kde=True, bins=30)
plt.title("Distribution of Tax Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


## Handling Missing Values for Categorical Variables

### Are there missing values? How many?
df.isna().sum()

### Are there missing values in `currency`?
df["currency"].isna().sum()
print(df["currency"].isna().sum())

### Calculate the mode for `currency`
df["currency"].mode()[0]
print(df["currency"].mode()[0])

### Replace the missing values in `currency` with the mode
df["currency"] = np.where(df["currency"].isnull(), df["currency"].mode()[0], df["currency"])

print(df["currency"].isna().sum())

### Outstanding proportion of missing value?
missing_values02 = df.isna().sum()
total_rows02 = len(df)
missing_value_proportion02 = missing_values02 / total_rows
print(missing_value_proportion02)
missing_value_percentage02 = missing_values02 * 100 / total_rows02
print(missing_value_percentage02.round(1))

### Filling missing values in `conversion_rate` with the category `Other`
df["conversion_rate"] = np.where(df["conversion_rate"].isnull(), "Other", df["conversion_rate"])


### Filling missing values in `document` with the category `Other`
df["document"] = np.where(df["document"].isnull(), "Other", df["document"])

### Filling missing values in 'operation_nature' with bfill
df["operation_nature"] = df["operation_nature"].fillna(method="bfill", inplace=True)

## Handling Missing Values That Do Not Appear Missing

### Checking for the '?' character in the 'credit_account' column (Method 1)
has_question_mark = df["credit_account"].isin(["?"]).any()
print(has_question_mark)

### Counting the frequency of each value in the 'credit_account' column (Method 2)
credit_account_counts = df["credit_account"].value_counts()
print(credit_account_counts)
#### Checking if '?' is in the counts and getting its number of occurrences
question_mark_count = credit_account_counts.get("?", 0)
print(question_mark_count)

### Identifying categorical columns (Method 3)
categorical_columns = df.select_dtypes(include=["object", "category"]).columns

### Check for the presence of '?' in each categorical column
for column in categorical_columns:
    has_question_mark = df[column].isin(["?"]).any()
    print(f"Does the column '{column}' contain '?'? {has_question_mark}")

### Replacing '?' with NaN and then filling missing values
df["credit_account"].replace("?", np.nan, inplace=True)
# df["credit_account"] = np.where(df["credit_account"].isin(["?"]), np.nan, df["credit_account"])

credit_account_counts02 = df["credit_account"].value_counts()
print(credit_account_counts02)

df["credit_account"].fillna(method="ffill", inplace=True)
# df["credit_account"] = np.where(df["credit_account"].isna(), df["credit_account"].fillna(method="ffill"), df["credit_account"])

credit_account_counts03 = df["credit_account"].value_counts()
print(credit_account_counts03)


