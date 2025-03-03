#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import realpath as realpath

# Load Datasets into DataFrames
path_to_insurance_data = "../000_common_dataset/us_health_insurance.csv"
path_to_nba_data = "../000_common_dataset/nba_player_salaries-2022-2023_season.csv"

real_path_to_insurance_data = realpath(path_to_insurance_data)
real_path_to_nba_data = realpath(path_to_nba_data)

df_insurance = pd.read_csv(real_path_to_insurance_data)
df_nba_salaries = pd.read_csv(real_path_to_nba_data)


def univariate_stats(df):
    """
    Generate descriptive statistics and visualisations for each feature in a DataFrame.

    This function computes and returns a DataFrame containing a variety of univariate
    statistics for each feature (column) in the input DataFrame `df`. It calculates
    metrics such as the data type, count of non-missing values, number of missing values,
    number of unique values, and mode for all features. For numerical features, it
    additionally calculates minimum, first quartile, median, third quartile, maximum,
    mean, standard deviation, skewness, and kurtosis. It also creates a histogram for
    numerical features and a count plot for categorical features.

    Parameters:
    - df (pd.DataFrame): The DataFrame for which univariate statistics are to be computed.

    Returns:
    - pd.DataFrame: A DataFrame where each row corresponds to a feature from the input
      DataFrame and columns contain the calculated statistics.
    """
    output_df = pd.DataFrame(
        columns=[
            "feature",
            "type",
            "count",
            "missing",
            "unique",
            "mode",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "mean",
            "std",
            "skew",
            "kurt",
        ]
    )
    output_df.set_index("feature", inplace=True)
    for col in df.columns:
        # Calculate metrics that apply to all columns dtypes
        dtype = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        mode = df[col].mode()[0]
        if pd.api.types.is_numeric_dtype(df[col]):
            # Calculate metrics that apply only to numerical features
            min = df[col].min()
            q1 = df[col].quantile(0.25)
            median = df[col].median()
            q3 = df[col].quantile(0.75)
            max = df[col].max()
            mean = df[col].mean()
            std = df[col].std()
            skew = df[col].skew()
            kurt = df[col].kurt()
            output_df.loc[col] = [
                dtype,
                count,
                missing,
                unique,
                mode,
                min,
                q1,
                median,
                q3,
                max,
                mean,
                std,
                skew,
                kurt,
            ]
            sns.histplot(data=df, x=col)
        else:
            output_df.loc[col] = [
                dtype,
                count,
                missing,
                unique,
                mode,
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
            ]
            sns.countplot(data=df, x=col)
        # print(f"Column: {col}")
        # print("dtype", "count", "missing", "unique", "mode")
        # print(dtype, count, missing, unique, mode)
        # print()
        plt.show()
    return output_df


univariate_stats(df_insurance)
univariate_stats(df_nba_salaries)
