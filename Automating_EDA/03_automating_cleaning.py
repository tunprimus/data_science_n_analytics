#!/usr/bin/env python3
import pandas as pd
from os.path import realpath as realpath

# Automating Data Cleaning

## Load Datasets into DataFrames

### Dataset with numerical labels
path_to_insurance_data = "../000_common_dataset/us_health_insurance.csv"
path_to_nba_data = "../000_common_dataset/nba_player_salaries-2022-2023_season.csv"
path_to_airbnb_data = "../000_common_dataset/airbnb-listings.csv"
### Dataset with categorical labels
path_to_airline_data = "../000_common_dataset/airline_passenger_satisfaction-train.csv"

real_path_to_insurance_data = realpath(path_to_insurance_data)
real_path_to_nba_data = realpath(path_to_nba_data)
real_path_to_airbnb_data = realpath(path_to_airbnb_data)
real_path_to_airline_data = realpath(path_to_airline_data)

df_insurance = pd.read_csv(real_path_to_insurance_data)
df_nba_salaries = pd.read_csv(real_path_to_nba_data)
df_airbnb = pd.read_csv(real_path_to_airbnb_data)
df_airline_satisfaction = pd.read_csv(real_path_to_airline_data)


## Basic Data Wrangling
##*********************##

### Eliminate Empty Columns, Columns with All Unique Values and Columns with Single Values

def basic_wrangling(df, features=[], missing_threshold=0.95, unique_threshold=0.95, messages=True):
    import pandas as pd

    all_cols = df.columns
    if len(features) == 0:
        features = all_cols
    for feat in features:
        if feat in all_cols:
            missing = df[feat].isna().sum()
            unique = df[feat].nunique()
            rows = df.shape[0]
            if (missing / rows) >= missing_threshold:
                if messages:
                    print(f"Too much missing ({missing} out of {rows} rows, {round(((missing / rows) * 100), 1)}%) for {feat}")
                df.drop(columns=[feat], inplace=True)
            elif (unique / rows) >= unique_threshold:
                if (df[feat].dtype in ["int64", "object"]):
                    if messages:
                        print(f"Too many unique values ({unique} out of {rows} rows, {round(((unique / rows) * 100), 1)}%) for {feat}")
                    df.drop(columns=[feat], inplace=True)
            elif unique == 1:
                if messages:
                    print(f"Only one value ({df[feat].unique()[0]}) for {feat}")
                df.drop(columns=[feat], inplace=True)
    else:
        if messages:
            print(f"The feature \'{feat}\' does not exist as spelled in the DataFrame provided.")
    return df



### Date and Time Management

def can_convert_dataframe_to_datetime(df, col_list=[], return_result=True, messages=False):
    """
    Check if columns in a DataFrame can be converted to datetime format.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to check
    col_list : list, default []
        List of columns to check. If empty, all columns with dtype of object will be checked.
    return_result : bool, default True
        If `True`, return a dictionary with column names as keys and boolean values indicating
        whether each column can be converted to datetime format. If `False`, return nothing and print
        messages.
    messages : bool, default False
        If `True`, print messages indicating whether each column can be converted to datetime format.

    Returns
    -------
    result : dict or list
        Dictionary with column names as keys and boolean values indicating
        whether each column can be converted to datetime format, or list of boolean values.
    """
    import pandas as pd
    import numpy as np
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    # Change message flag based on whether or not to return result
    if not return_result:
        messages = True

    length_col_list = len(col_list)
    # Define type of return object
    if length_col_list == 1:
        result = []
    else:
        result = {}
    # Determine how many columns to use in DataFrame
    if length_col_list == 0:
        columns_to_check = df.columns
    else:
        columns_to_check = df.columns[df.columns.isin(col_list)]
    # Check only columns with dtype of object
    for col in columns_to_check:
        # result = []
        if df[col].dtype == "object":
            can_be_datetime = False
            try:
                df_flt_tmp = df[col].astype(np.float64)
                can_be_datetime = False
            except:
                try:
                    df_dt_tmp = pd.to_datetime(df[col])
                    can_be_datetime = is_datetime(df_dt_tmp)
                except:
                    pass
            if messages:
                print(f"Can convert {col} to datetime? {can_be_datetime}")
            # Choose return data structure
            if length_col_list == 1:
                result.append(can_be_datetime)
            else:
                result[col] = can_be_datetime
    if return_result:
        return result



def batch_convert_to_datetime(df, split_datetime=True, add_hr_min_sec=False, days_to_today=False, drop_date=True, messages=True):
    """
    Convert object columns in a DataFrame to datetime format if possible.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to convert
    split_datetime : bool, default True
        If `True`, split datetime columns into separate columns for year, month, day and weekday.
    add_hr_min_sec : bool, default False
        If `True`, add separate columns for hour, minute and second.
    days_to_today : bool, default False
        If `True`, add a column with the number of days to today.
    drop_date : bool, default True
        If `True`, drop the original column after conversion.
    messages : bool, default True
        If `True`, print messages indicating which columns were converted.

    Returns
    -------
    df : pandas DataFrame
        DataFrame with converted datetime columns.
    """
    import pandas as pd
    import numpy as np
    import sys
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    for col in df.columns[df.dtypes == "object"]:
        try:
            df_dt_tmp = pd.to_datetime(df[col])
            try:
                df_flt_tmp = df[col].astype(np.float64)
                if messages:
                    print(f"Warning, NOT converting column '{col}', because it is ALSO convertible to float64.", file=sys.stderr)
            except:
                df[col] = df_dt_tmp
                if split_datetime:
                    df[f"{col}_datetime"] = df[col]
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_weekday"] = df[col].dt.day_name()
                if add_hr_min_sec:
                    df[f"{col}_hour"] = df[col].dt.hour
                    df[f"{col}_minute"] = df[col].dt.minute
                    df[f"{col}_second"] = df[col].dt.second
                if days_to_today:
                    df[f"{col}_days_to_today"] = (pd.to_datetime("now") - df[col]).dt.days
                if drop_date:
                    df.drop(columns=[col], inplace=True)
                if messages:
                    print(f"FYI, converted column '{col}' to datetime.", file=sys.stderr)
                    print(f"Is '{df[col]}' now datetime? {is_datetime(df[col])}")
        # Cannot convert some elements of the column to datetime...
        except:
            pass
    return df


def parse_date(df, features=[], days_to_today=False, drop_date=True, messages=True):
    """
    Parse specified date features in a DataFrame and extract related information.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the features to be parsed.
    features : list, default []
        List of column names to parse as date features. If empty, no columns are parsed.
    days_to_today : bool, default False
        If `True`, calculate the number of days from each date to today's date.
    drop_date : bool, default True
        If `True`, drop the original date columns after parsing.
    messages : bool, default True
        If `True`, print messages indicating parsing status for each feature.

    Returns
    -------
    df : pandas DataFrame
        DataFrame with parsed date features and additional extracted information.
    """
    import pandas as pd
    from datetime import datetime as pydt

    all_cols = df.columns
    for feat in features:
        if feat in all_cols:
            try:
                df[feat] = pd.to_datetime(df[feat])
                df[f"{feat}_datetime"] = df[feat]
                df[f"{feat}_year"] = df[feat].dt.year
                df[f"{feat}_month"] = df[feat].dt.month
                df[f"{feat}_day"] = df[feat].dt.day
                df[f"{feat}_weekday"] = df[feat].dt.day_name()
                if days_to_today:
                    df[f"{feat}_days_until_today"] = (pydt.today() - df[feat]).dt.days
                if drop_date:
                    df.drop(columns=[feat], inplace=True)
            except:
                if messages:
                    print(f"Could not convert feature \'{feat}\' to datetime.")
        else:
            if messages:
                print(f"Feature \'{feat}\' does not exist as spelled in the DataFrame provided.")
    return df

### Bin Low Count Groups Values


def bin_categories(df, features=[], cutoff=0.05, replace_with="Other", messages=True):
    """
    Bins low count groups values into one category

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to clean
    features : list of str, default []
        Columns to consider for binning
    cutoff : float, default 0.05
        Proportion of total samples to define low count groups
    replace_with : str, default 'Other'
        Value to replace low count groups with
    messages : bool, default True
        If `True`, print messages indicating which columns were cleaned

    Returns
    -------
    df : pandas DataFrame
        DataFrame with low count groups binned
    """
    import pandas as pd
    for feat in features:
        if feat in df.columns:
            if not pd.api.types.is_numeric_dtype(df[feat]):
                other_list = df[feat].value_counts()[(df[feat].value_counts() / df.shape[0]) < cutoff].index
                df.loc[df[feat].isin(other_list), feat] = replace_with
        else:
            if messages:
                print(f"The feature \'{feat}\' does not exist in the DataFrame provided. No binning performed.")
    return df


## Outliers
##*********************##

### Traditional One-at-a-time Methods

def clean_outlier(df, features=[], skew_threshold=1, messages=True):
    import pandas as pd
    import numpy as np
    for feat in features:
        if feat in df.columns:
            if pd.api.types.is_numeric_dtype(df[feat]):
                if len(df[feat].unique()) == 1:
                    if not all(df[feat].value_counts().index.isin([0, 1])):
                        # Empirical rule
                        pass
                        # Tukey boxplot rule
                        pass
                    else:
                        if messages:
                            print(f"Feature \'{feat}\' is dummy coded (0, 1) and was ignored.")
                else:
                    if messages:
                        print(f"Feature \'{feat}\' has only one unique value ({df[feat].unique()[0]}).")
            else:
                if messages:
                    print(f"Feature \'{feat}\' is categorical and was ignored.")
        else:
            if messages:
                print(f"Feature \'{feat}\' does not exist in the DataFrame provided. No outlier removal performed.")
    return df

### Newer All-at-once Methods Based on Clustering





## Skewness
##*********************##





## Missing Data
##*********************##

### Missing Not At Random (MNAR)




### Missing Completely At Random (MCAR)





### Missing At Random (MAR)






#--------#
# TESTS  #
#--------#

basic_wrangling(df_insurance)
basic_wrangling(df_nba_salaries)
basic_wrangling(df_airbnb)
basic_wrangling(df_airline_satisfaction)

can_convert_dataframe_to_datetime(df_airbnb)
can_convert_dataframe_to_datetime(df_airbnb, col_list=["name"])
can_convert_dataframe_to_datetime(df_airbnb, col_list=["name"])[0]
can_convert_dataframe_to_datetime(df_airbnb, col_list=["last_review"])[0]
can_convert_dataframe_to_datetime(df_airbnb, col_list=["name", "last_review", "host_name"])


batch_convert_to_datetime(df_airbnb)

parse_date(df_airbnb, features=["last_review"])
parse_date(df_airbnb, features=["last_review"], days_to_today=True)


bin_categories(df_airbnb, features=["neighbourhood"], cutoff=0.025)


