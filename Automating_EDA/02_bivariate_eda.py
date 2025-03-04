#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def scatterplot(df, feature, label, num_dp=4, linecolour="darkorange"):
    """
    Creates a scatterplot between two features in a DataFrame, with a regression line included.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data
    feature : str
        The feature to plot on the x-axis
    label : str
        The feature to plot on the y-axis
    num_dp : int
        The number of decimal places to round the regression equation to
    linecolour : str
        The colour of the regression line

    Returns
    -------
    None
    """
    # Create the plot
    # sns.scatterplot(x=df[feature], y=df[label])
    sns.regplot(x=df[feature], y=df[label], line_kws={"color": linecolour})
    # Calculate the regression line
    results = stats.linregress(df[feature], df[label])
    slope = results.slope
    intercept = results.intercept
    r = results.rvalue
    p = results.pvalue
    stderr = results.stderr
    intercept_stderr = results.intercept_stderr
    text_str = f"y = {round(slope, num_dp)}x + {round(intercept, num_dp)}\n"
    text_str += f"r = {round(r, num_dp)}\n"
    text_str += f"p = {round(p, num_dp)}"
    # Annotations
    plt.text(0.95, 0.2, text_str, fontsize=12, transform=plt.gcf().transFigure)
    # Show plot
    plt.show()


def bar_chart(df, feature, label, num_dp=4):
    """
    Create a bar chart of feature by label and calculate the ANOVA between the different levels of feature.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data
    feature : str
        The feature to plot on the x-axis
    label : str
        The feature to plot on the y-axis
    num_dp : int
        The number of decimal places to round the ANOVA results to

    Returns
    -------
    None
    """
    # Create the plot
    sns.barplot(x=df[feature], y=df[label])
    # Create the numerical lists to calculate the ANOVA
    groups = df[feature].unique()
    # print(groups)
    group_lists = []
    for g in groups:
        n_list = df[df[feature] == g][label]
        group_lists.append(n_list)
    F, p = stats.f_oneway(*group_lists)
    text_str = f"F: {round(F, num_dp)}\n"
    text_str += f"p: {round(p, num_dp)}"
    # If there are too many feature groups, print x labels vertically
    if df[feature].nunique() > 7:
        plt.xticks(rotation=90)
    # Annotations
    plt.text(0.95, 0.2, text_str, fontsize=12, transform=plt.gcf().transFigure)
    # Show plot
    plt.show()


def crosstab(df, feature, label, num_dp=4):
    """
    Creates a heatmap of a contingency table between two categorical features in a DataFrame and calculates the Chi-Squared statistic.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data
    feature : str
        The feature to plot on the x-axis
    label : str
        The feature to plot on the y-axis
    num_dp : int
        The number of decimal places to round the Chi-Squared results to

    Returns
    -------
    None
    """
    contingency_table = pd.crosstab(df[feature], df[label])
    results = stats.chi2_contingency(contingency_table)
    X2 = results.statistic
    p = results.pvalue
    dof = results.dof
    expected_freq = results.expected_freq
    text_str = f"X2: {round(X2, num_dp)}\n"
    text_str += f"p: {round(p, num_dp)}\n"
    text_str += f"dof: {round(dof, num_dp)}"
    # Annotations
    plt.text(0.95, 0.2, text_str, fontsize=12, transform=plt.gcf().transFigure)
    # Generate heatmap
    ct_df = pd.DataFrame(np.rint(expected_freq).astype("int64"), columns=contingency_table.columns, index=contingency_table.index)
    sns.heatmap(ct_df, annot=True, fmt="d", cmap="coolwarm")
    # Show plot
    plt.show()


def bivariate_stats(df, label, num_dp=4):
    output_df = pd.DataFrame(columns=["missing", "missing_%", "skew", "type", "num_unique", "p", "r", "tau", "rho", "y = m(x) + b", "F", "X2"])

    for feature in df.columns:
        if feature != label:
            # Calculate statistics that apply to all datatypes
            df_temp = df[[feature, label]].copy()
            df_temp = df_temp.dropna().copy()
            missing = (df.shape[0] - df_temp.shape[0])
            buffer = ((df.shape[0] - df_temp.shape[0]) / df.shape[0])
            missing_pct = round(buffer * 100, num_dp)
            dtype = df_temp[feature].dtype
            num_unique = df_temp[feature].nunique()
            if (pd.api.types.is_numeric_dtype(df_temp[feature])) and (pd.api.types.is_numeric_dtype(df_temp[label])):
                # Process N2N relationships
                ## Pearson linear regression
                results_p = stats.linregress(df_temp[feature], df_temp[label])
                slope = results_p.slope
                intercept = results_p.intercept
                r = results_p.rvalue
                p = results_p.pvalue
                stderr = results_p.stderr
                intercept_stderr = results_p.intercept_stderr
                ## Other linear regressions
                results_k = stats.kendalltau(df_temp[feature], df_temp[label])
                tau = results_k.statistic
                tp = results_k.pvalue
                results_r = stats.spearmanr(df_temp[feature], df_temp[label])
                rho = results_r.statistic
                rp = results_r.pvalue
                ## Skew
                skew = round((df_temp[feature].skew()), num_dp)
                output_df.loc[feature] = [missing, f"{missing_pct}%", skew, dtype, num_unique, round(p, num_dp), round(r, num_dp), round(tau, num_dp), round(rho, num_dp), f"y = {round(slope, num_dp)}x + {round(intercept, num_dp)}", "--", "--"]
                scatterplot(df_temp, feature, label)
            elif not(pd.api.types.is_numeric_dtype(df_temp[feature])) and not(pd.api.types.is_numeric_dtype(df_temp[label])):
                # Process C2C relationships
                contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
                results = stats.chi2_contingency(contingency_table)
                X2 = results.statistic
                p = results.pvalue
                dof = results.dof
                expected_freq = results.expected_freq
                output_df.loc[feature] = [missing, f"{missing_pct}%", "--", dtype, num_unique, round(p, num_dp), "--", "--", "--", "--", "--", round(X2, num_dp)]
                crosstab(df_temp, feature, label)
            else:
                # Process C2N and N2C relationships
                if pd.api.types.is_numeric_dtype(df_temp[feature]):
                    num = feature
                    cat = label
                    skew = round((df_temp[feature].skew()), num_dp)
                else:
                    num = label
                    cat = feature
                    skew = "--"
                groups = df_temp[cat].unique()
                group_lists = []
                for g in groups:
                    n_list = df_temp[df_temp[cat] == g][num]
                    group_lists.append(n_list)
                F, p = stats.f_oneway(*group_lists)
                output_df.loc[feature] = [missing, f"{missing_pct}%", skew, dtype, num_unique, round(p, num_dp), "--", "--", "--", "--", round(F, num_dp), "--"]
                bar_chart(df_temp, cat, num)
    try:
        return output_df.sort_values(by="p", ascending=True)
    except Exception:
        return output_df



bivariate_stats(df_insurance, "charges")
bivariate_stats(df_nba_salaries, "Salary")
bivariate_stats(df_airline_satisfaction, "satisfaction")

scatterplot(df_insurance, "age", "charges")
bar_chart(df_insurance, "smoker", "charges")
bar_chart(df_insurance, "region", "charges")
crosstab(df_airline_satisfaction, "Gender", "satisfaction")
crosstab(df_airline_satisfaction, "Class", "satisfaction")

