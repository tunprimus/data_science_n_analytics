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
        print(columns_to_check)
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


def can_convert_value_to_datetime(val, messages=False):
    """
    Check if a value can be converted to datetime format.

    Parameters
    ----------
    val : value to check
    messages : bool, default False
        If `True`, print messages indicating whether the value can be converted to datetime format.

    Returns
    -------
    can_be_datetime : bool
        Whether the value can be converted to datetime format.
    """
    from pandas.api.types import is_datetime64_any_dtype as is_datetime
    try:
        df_dt_tmp = pd.to_datetime(val)
        try:
            df_flt_tmp = val.astype(np.float64)
            can_be_datetime = False
        except:
            can_be_datetime = is_datetime(df_dt_tmp)
    except:
        can_be_datetime = False
    if messages:
        print(f"Can convert {val} to datetime? {can_be_datetime}")
    return can_be_datetime



def convert_to_datetime(df, messages=True):
    """
    Convert columns of object dtype to datetime, if possible.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to check
    messages : bool, default True
        If `True`, print messages indicating which columns were converted to datetime format.

    Returns
    -------
    df : pandas DataFrame
        DataFrame with datetime columns, if possible
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
                if messages:
                    print(f"FYI, converted column '{col}' to datetime.", file=sys.stderr)
                    print(f"Is '{df[col]}' now datetime? {is_datetime(df[col])}")
        # Cannot convert some elements of the column to datetime...
        except:
            pass
    return df


def parse_date(df, features=[], days_to_today=False, drop_date=True, messages=True):
    import pandas as pd
    from datetime import datetime as pydt
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    all_cols = df.columns
    for feat in features:
        if (feat in all_cols) and (can_convert_value_to_datetime(df[feat])):
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
        else:
            if messages:
                print(f"The feature \'{feat}\' does not exist as spelled in the DataFrame provided.")
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

can_convert_value_to_datetime(df_airbnb["name"])
can_convert_value_to_datetime(df_airbnb["last_review"])
can_convert_value_to_datetime(df_airbnb["host_id"])

convert_to_datetime(df_airbnb)

parse_date(df_airbnb, features=["last_review"])
parse_date(df_airbnb, features=["last_review"], days_to_today=True)


bin_categories(df_airbnb, features=["neighbourhood"], cutoff=0.025)


