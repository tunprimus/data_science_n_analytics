#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import realpath as realpath

path_to_insurance_data = "../000_common_dataset/insurance.csv"
path_to_nba_data = "../000_common_dataset/nba_player_salaries-2022-2023_season.csv"

real_path_to_insurance_data = realpath(path_to_insurance_data)
real_path_to_nba_data = realpath(path_to_nba_data)
df_insurance = pd.read_csv(real_path_to_insurance_data)
df_nba = pd.read_csv(real_path_to_nba_data)


def univariate(df):
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
    for col in df:
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
        # print(f"Column: {col}")
        # print("dtype", "count", "missing", "unique", "mode")
        # print(dtype, count, missing, unique, mode)
        # print()
    return output_df


univariate(df_insurance)
univariate(df_nba)
