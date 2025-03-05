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

def parse_date(df, features=[], days_to_today=False, drop_date=True, messages=True):
    import pandas as pd
    from datetime import datetime as pydt

    all_cols = df.columns
    for feat in features:
        if feat in all_cols:
            df[feat] = pd.to_datetime(df[feat])
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

parse_date(df_airbnb, features=["last_review"])
parse_date(df_airbnb, features=["last_review"], days_to_today=True)
