#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import realpath as realpath
from scipy import stats


# Load Datasets into DataFrames
path_to_insurance_data = "../000_common_dataset/us_health_insurance.csv"
path_to_nba_data = "../000_common_dataset/nba_player_salaries-2022-2023_season.csv"
path_to_airline_data = "../000_common_dataset/airline_passenger_satisfaction-train.csv"

real_path_to_insurance_data = realpath(path_to_insurance_data)
real_path_to_nba_data = realpath(path_to_nba_data)
real_path_to_airline_data = realpath(path_to_airline_data)

df_insurance = pd.read_csv(real_path_to_insurance_data)
df_nba_salaries = pd.read_csv(real_path_to_nba_data)
df_airline_satisfaction = pd.read_csv(real_path_to_airline_data)


def bivariate_stats(df, label, round_to=4):
    output_df = pd.DataFrame(columns=["p", "r", "y = m(x) + b", "F"])

    for feature in df.columns:
        if feature != label:
            if (pd.api.types.is_numeric_dtype(df[feature])) and (pd.api.types.is_numeric_dtype(df[label])):
                # Process N2N relationships
                results = stats.linregress(df[feature], df[label])
                slope = results.slope
                intercept = results.intercept
                r = results.rvalue
                p = results.pvalue
                stderr = results.stderr
                intercept_stderr = results.intercept_stderr
                output_df.loc[feature] = [round(p, round_to), round(r, round_to), f"y = {round(slope, round_to)}x + {round(intercept, round_to)}", "--"]
            elif not(pd.api.types.is_numeric_dtype(df[feature])) and not(pd.api.types.is_numeric_dtype(df[label])):
                # Process C2C relationships
                output_df.loc[feature] = ["--", "--", "--", "--"]
            else:
                # Process C2N and N2C relationships
                if pd.api.types.is_numeric_dtype(df[feature]):
                    num = feature
                    cat = label
                else:
                    num = label
                    cat = feature
                # print(f"Cat: {cat} | Num: {num}")
                groups = df[cat].unique()
                # print(groups)
                group_lists = []
                for g in groups:
                    n_list = df[df[cat] == g][num]
                    group_lists.append(n_list)
                F, p = stats.f_oneway(*group_lists)
                # output
                output_df.loc[feature] = [round(p, round_to), "--", "--", round(F, round_to)]
    # return output_df.sort_values(by="r", ascending=False)
    return output_df



bivariate_stats(df_insurance, "charges")
bivariate_stats(df_nba_salaries, "Salary")
bivariate_stats(df_airline_satisfaction, "satisfaction")

