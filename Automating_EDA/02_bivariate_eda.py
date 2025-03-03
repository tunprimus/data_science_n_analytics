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


def bivariate_stats(df, label, num_dp=4):
    output_df = pd.DataFrame(columns=["missing", "missing_%", "p", "r", "y = m(x) + b", "F", "X2"])

    for feature in df.columns:
        if feature != label:
            df_temp = df[[feature, label]].copy()
            df_temp = df_temp.dropna().copy()
            buffer = ((df.shape[0] - df_temp.shape[0]) / df.shape[0])
            missing = round(buffer, num_dp)
            missing_pct = round(buffer * 100, num_dp)
            if (pd.api.types.is_numeric_dtype(df_temp[feature])) and (pd.api.types.is_numeric_dtype(df_temp[label])):
                # Process N2N relationships
                results = stats.linregress(df_temp[feature], df_temp[label])
                slope = results.slope
                intercept = results.intercept
                r = results.rvalue
                p = results.pvalue
                stderr = results.stderr
                intercept_stderr = results.intercept_stderr
                output_df.loc[feature] = [missing, f"{missing_pct}%", round(p, num_dp), round(r, num_dp), f"y = {round(slope, num_dp)}x + {round(intercept, num_dp)}", "--", "--"]
            elif not(pd.api.types.is_numeric_dtype(df_temp[feature])) and not(pd.api.types.is_numeric_dtype(df_temp[label])):
                # Process C2C relationships
                contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
                results = stats.chi2_contingency(contingency_table)
                X2 = results.statistic
                p = results.pvalue
                dof = results.dof
                expected_freq = results.expected_freq
                output_df.loc[feature] = [missing, f"{missing_pct}%", round(p, num_dp), "--", "--", "--", round(X2, num_dp)]
            else:
                # Process C2N and N2C relationships
                if pd.api.types.is_numeric_dtype(df_temp[feature]):
                    num = feature
                    cat = label
                else:
                    num = label
                    cat = feature
                # print(f"Cat: {cat} | Num: {num}")
                groups = df_temp[cat].unique()
                # print(groups)
                group_lists = []
                for g in groups:
                    n_list = df_temp[df_temp[cat] == g][num]
                    group_lists.append(n_list)
                F, p = stats.f_oneway(*group_lists)
                # output
                output_df.loc[feature] = [missing, f"{missing_pct}%", round(p, num_dp), "--", "--", round(F, num_dp), "--"]
    try:
        return output_df.sort_values(by="p", ascending=True)
    except Exception:
        return output_df



bivariate_stats(df_insurance, "charges")
bivariate_stats(df_nba_salaries, "Salary")
bivariate_stats(df_airline_satisfaction, "satisfaction")

