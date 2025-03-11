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


## Some Constants
##*********************##
RANDOM_SEED = 42
RANDOM_SAMPLE_SIZE = 13
NUM_DEC_PLACES = 4
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72


## Some Useful Functions
##*********************##

def cpu_logical_cores_count():
    """
    Return the number of logical cores on the machine.

    The number of logical cores is the number of physical cores times the
    number of threads that can run on each core (Simultaneous Multithreading,
    SMT). If the number of logical cores cannot be determined, an exception is
    raised.
    """
    import joblib
    import multiprocessing
    import os
    import psutil
    import re
    import subprocess
    # For Python 2.6+
    # Using multiprocessing module
    try:
        n_log_cores = multiprocessing.cpu_count()
        if n_log_cores > 0:
            return n_log_cores
    except (ImportError, NotImplementedError):
        pass
    # Using joblib module
    try:
        n_log_cores = joblib.cpu_count()
        if n_log_cores > 0:
            return n_log_cores
    except (ImportError, NotImplementedError):
        pass
    # Using psutil module
    try:
        n_log_cores = psutil.cpu_count()
        if n_log_cores > 0:
            return n_log_cores
    except (ImportError, AttributeError):
        pass
    # Using os module
    try:
        n_log_cores = os.cpu_count()
        if n_log_cores is None:
            raise NotImplementedError
        if n_log_cores > 0:
            return n_log_cores
    except:
        pass
    # Check proc process
    try:
        m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", open("/proc/self/status").read())
        if m:
            res = bin(int(m.group(1).replace(",", "")))
            if res > 0:
                n_log_cores = res
                return n_log_cores
    except IOError:
        pass
    # POSIX
    try:
        res = int(os.sysconf("SC_NPROCESSORS_ONLN"))
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except (AttributeError, ValueError):
        pass
    # Windows
    try:
        res = int(os.environ["NUMBER_OF_PROCESSORS"])
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except (AttributeError, ValueError):
        pass
    # Jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except ImportError:
        pass
    # BSD
    try:
        sysctl = subprocess.Popen(["sysctl", "-n", "hw.ncpu"], stdout=subprocess.PIPE)
        sc_stdout = sysctl.communicate()[0]
        res = int(sc_stdout)
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except (OSError, ValueError):
        pass
    # Linux
    try:
        with (open("/proc/cpuinfo")) as fin:
            res = fin.read().count("processor\t:")
            if res > 0:
                n_log_cores = res
                return n_log_cores
    except IOError:
        pass
    # Solaris
    try:
        pseudo_devices = os.listdir("/dev/pseudo")
        res = 0
        for pd in pseudo_devices:
            if re.match(r"^cpuid@[0-9]+$", pd):
                res += 1
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except OSError:
        pass
    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open("/var/run/dmesg.boot").read()
        except IOError:
            dmesg_process = subprocess.Popen(["dmesg"], stdout=subprocess.PIPE)
            dmesg = dmesg_process.communicate()[0]
        res = 0
        while "\ncpu" + str(res) + ":" in dmesg:
            res += 1
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except OSError:
        pass
    raise Exception("Cannot determine number of cores on this system.")


LOGICAL_CORES = cpu_logical_cores_count()


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


def parse_column_as_date(df, features=[], days_to_today=False, drop_date=True, messages=True):
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

def clean_outlier_per_column(df, features=[], skew_threshold=1, handle_outliers="remove", num_dp=4, messages=True):
    """
    Clean outliers from a column in a DataFrame

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to clean
    features : list of str, default []
        Columns to consider for outlier cleaning
    skew_threshold : float, default 1
        Threshold to determine if a column is skewed
    handle_outliers : str, default 'remove'
        How to handle outliers. Options are 'remove', 'replace', 'impute', 'null'
    num_dp : int, default 4
        Number of decimal places to round to
    messages : bool, default True
        If `True`, print messages indicating which columns were cleaned

    Returns
    -------
    df : pandas DataFrame
        DataFrame with outliers cleaned
    """
    import pandas as pd
    import numpy as np
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    # Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
    np.float = np.float64
    np.int = np.int_
    np.object = np.object_
    np.bool = np.bool_

    pd.set_option("mode.copy_on_write", True)

    for feat in features:
        if feat in df.columns:
            if pd.api.types.is_numeric_dtype(df[feat]):
                if len(df[feat].unique()) != 1:
                    if not all(df[feat].value_counts().index.isin([0, 1])):
                        skew = df[feat].skew()
                        # Tukey boxplot rule
                        if abs(skew) > skew_threshold:
                            q1 = df[feat].quantile(0.25)
                            q3 = df[feat].quantile(0.75)
                            iqr = q3 - q1
                            lo_bound = q1 - 1.5 * iqr
                            hi_bound = q3 + 1.5 * iqr
                        # Empirical rule
                        else:
                            lo_bound = df[feat].mean() - (3 * df[feat].std())
                            hi_bound = df[feat].mean() + (3 * df[feat].std())
                        # Get the number of outlier data points
                        min_count = df.loc[df[feat] < lo_bound, feat].shape[0]
                        max_count = df.loc[df[feat] > hi_bound, feat].shape[0]
                        if (min_count > 0) or (max_count > 0):
                            # Remove rows with the outliers
                            if handle_outliers == "remove":
                                df = df[(df[feat] >= lo_bound) & (df[feat] <= hi_bound)]
                            # Replace outliers with min-max cutoff
                            elif handle_outliers == "replace":
                                df.loc[df[feat] < lo_bound, feat] = lo_bound
                                df.loc[df[feat] > hi_bound, feat] = hi_bound
                            # Impute with linear regression after deleting
                            elif handle_outliers == "impute":
                                df.loc[df[feat] < lo_bound, feat] = np.nan
                                df.loc[df[feat] > hi_bound, feat] = np.nan
                                imputer = IterativeImputer(max_iter=10)
                                df_temp = df.copy()
                                df_temp = bin_categories(df_temp, features=df_temp.columns, messages=False)
                                df_temp = basic_wrangling(df_temp, features=df_temp.columns, messages=False)
                                df_temp = pd.get_dummies(df_temp, drop_first=True)
                                df_temp = pd.DataFrame(imputer.fit_transform(df_temp), columns=df_temp.columns, index=df_temp.index, dtype="float")
                                df_temp.columns = df_temp.columns.get_level_values(0)
                                df_temp.index = df_temp.index.astype("int64")
                                # Save only the column from df_temp being iterated on
                                df[feat] = df_temp[feat]
                            # Replace with null
                            elif handle_outliers == "null":
                                df.loc[df[feat] < lo_bound, feat] = np.nan
                                df.loc[df[feat] > hi_bound, feat] = np.nan
                        if messages:
                            print(f"Feature \'{feat}\' has {min_count} value(s) below the lower bound ({round(lo_bound, num_dp)}) and {max_count} value(s) above the upper bound ({round(hi_bound, num_dp)}).")
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


def clean_outlier_by_all_columns(df, drop_proportion=0.013, distance_method="manhattan", min_samples=5, num_dp=4, num_cores_for_dbscan=LOGICAL_CORES-2 if LOGICAL_CORES > 3 else 1, messages=True):
    """
    Clean outliers from a DataFrame based on a range of eps values.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to clean
    drop_proportion : float, default 0.013
        Proportion of total samples to define outliers
    distance_method : str, default "manhattan"
        Distance method to use in DBSCAN
    min_samples : int, default 5
        Minimum samples to form a dense region
    num_dp : int, default 4
        Number of decimal places to round to
    num_cores_for_dbscan : int, default LOGICAL_CORES-2 if LOGICAL_CORES > 3 else 1
        Number of cores to use in DBSCAN
    messages : bool, default True
        If `True`, print messages indicating which columns were cleaned

    Returns
    -------
    df : pandas DataFrame
        DataFrame with outliers cleaned
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import time
    from sklearn import preprocessing
    from sklearn.cluster import DBSCAN

    # Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
    np.float = np.float64
    np.int = np.int_
    np.object = np.object_
    np.bool = np.bool_

    pd.set_option("mode.copy_on_write", True)

    # Clean the DataFrame first
    num_cols_with_missing_values = df.shape[1] - df.dropna(axis="columns").shape[1]
    df.dropna(axis="columns", inplace=True)
    if messages:
        print(f"{num_cols_with_missing_values} column(s) with missing values dropped.")
    num_rows_with_missing_values = df.shape[0] - df.dropna(axis="columns").shape[0]
    df.dropna(inplace=True)
    if messages:
        print(f"{num_rows_with_missing_values} row(s) with missing values dropped.")
    # Handle basic wrangling, binning and outliers
    df_temp = df.copy()
    df_temp = bin_categories(df_temp, features=df_temp.columns, messages=False)
    df_temp = basic_wrangling(df_temp, features=df_temp.columns, messages=False)
    df_temp = pd.get_dummies(df_temp, drop_first=True)
    # Normalise the data
    df_temp = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df_temp), columns=df_temp.columns, index=df_temp.index)
    # Calculate outliers based on a range of eps values
    outliers_per_eps = []
    outliers_per_eps_history = {}
    outliers = df_temp.shape[0]
    eps_loop = 0
    counter = 0
    row_count = df_temp.shape[0]
    if row_count < 500:
        INCREMENT_VAL = 0.010
    elif row_count < 1000:
        INCREMENT_VAL = 0.025
    elif row_count < 2000:
        INCREMENT_VAL = 0.050
    elif row_count < 10000:
        INCREMENT_VAL = 0.075
    elif row_count < 25000:
        INCREMENT_VAL = 0.100
    elif row_count < 50000:
        INCREMENT_VAL = 0.200
    elif row_count < 100000:
        INCREMENT_VAL = 0.250
    elif row_count < 250000:
        INCREMENT_VAL = 0.350
    else:
        INCREMENT_VAL = 0.500
    db_scan_time_start = time.time_ns()
    while outliers > 0:
        loop_start_time = time.time_ns()
        eps_loop += INCREMENT_VAL
        db_loop = DBSCAN(eps=eps_loop, metric=distance_method, min_samples=min_samples, n_jobs=num_cores_for_dbscan).fit(df_temp)
        outliers = np.count_nonzero(db_loop.labels_ == -1)
        outliers_per_eps.append(outliers)
        outliers_percent = (outliers / row_count) * 100
        outliers_per_eps_history[f"{counter}_trial"] = {}
        outliers_per_eps_history[f"{counter}_trial"]["eps_val"] = round(eps_loop, num_dp)
        outliers_per_eps_history[f"{counter}_trial"]["outliers"] = outliers
        outliers_per_eps_history[f"{counter}_trial"]["outliers_percent"] = round(outliers_percent, num_dp)
        loop_end_time = time.time_ns()
        loop_time_diff_ns = (loop_end_time - loop_start_time)
        loop_time_diff_s = (loop_end_time - loop_start_time) / 1000000000
        outliers_per_eps_history[f"{counter}_trial"]["loop_duration_ns"] = loop_time_diff_ns
        outliers_per_eps_history[f"{counter}_trial"]["loop_duration_s"] = loop_time_diff_s
        counter += 1
        if messages:
            print(f"eps = {round(eps_loop, num_dp)}, (outliers: {outliers}, percent: {round(outliers_percent, num_dp)}% in {round(loop_time_diff_s, num_dp)}s)")
    to_drop = min(outliers_per_eps, key=lambda x: abs(x - round((drop_percent * row_count), num_dp)))
    # Find the optimal eps value
    eps = (outliers_per_eps.index(to_drop) + 1) * INCREMENT_VAL
    outliers_per_eps_history["optimal_eps"] = eps
    db_scan_time_end = time.time_ns()
    db_scan_time_diff_s = (db_scan_time_end - db_scan_time_start) / 1000000000
    outliers_per_eps_history["db_scan_duration_s"] = db_scan_time_diff_s
    outliers_per_eps_history["distance_metric_used"] = distance_method
    outliers_per_eps_history["min_samples_used"] = min_samples
    outliers_per_eps_history["drop_proportion_used"] = drop_proportion
    outliers_per_eps_history["timestamp"] = pd.Timestamp.now()
    if messages:
        print(f"Optimal eps value: {round(eps, num_dp)}")
        # print(f"History: {outliers_per_eps_history}")
        print(f"\nHistory:")
        for key01, val01 in outliers_per_eps_history.items():
            if not isinstance(val01, dict):
                if isinstance(val01, (int, float)):
                    print(f"{key01}: {round(val01, num_dp)}")
                else:
                    print(f"{key01}: {val01}")
                continue
            else:
                print(f"{key01}")
                for key02, val02 in val01.items():
                    print(f"{key02}: {round(val02, num_dp)}")
                print("*********************")
            print()
    db = DBSCAN(eps=eps, metric=distance_method, min_samples=min_samples, n_jobs=num_cores_for_dbscan).fit(df_temp)
    df["outlier"] = db.labels_
    if messages:
        print(f"{df[df['outlier'] == -1].shape[0]} row(s) of outliers found for removal.")
        sns.lineplot(x=range(1, len(outliers_per_eps) + 1), y=outliers_per_eps)
        sns.scatterplot(x=[eps/INCREMENT_VAL], y=[to_drop])
        plt.xlabel(f"eps (divided by {INCREMENT_VAL})")
        plt.ylabel("Number of outliers")
        plt.show()
    # Drop rows that are outliers
    df = df[df["outlier"] != -1]
    # df.drop("outlier", axis="columns", inplace=True)
    return df



## Skewness
##*********************##


def skew_correct(df, feature, max_power=103, messages=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from os.path import realpath as realpath

    # Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
    np.float = np.float64
    np.int = np.int_
    np.object = np.object_
    np.bool = np.bool_

    pd.set_option("mode.copy_on_write", True)
    GOLDEN_RATIO = 1.618033989
    FIG_WIDTH = 20
    FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
    FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
    FIG_DPI = 72

    # rcParams for Plotting
    plt.rcParams["figure.figsize"] = FIG_SIZE
    plt.rcParams["figure.dpi"] = FIG_DPI

    # Check to use only numerical features
    if not pd.api.types.is_numeric_dtype(df[feature]):
        if messages:
            print(f"The feature \'{feature}\' is not numerical. No transformation performed.")
        return df

    # Clean out missing data
    df = basic_wrangling(df, messages=False)
    if messages:
        print(f"{df.shape[0] - df.dropna().shape[0]} row(s) with missing values dropped.")
    df.dropna(inplace=True)

    # In case the dataset is too big, use a subsample
    df_temp = df.copy()
    if df_temp.memory_usage(deep=True).sum() > 1_000_000:
        df_temp = df.sample(frac=round((5000 / df_temp.shape[0]), 2))

    # Identify the proper transformation exponent
    exp_val = 1
    skew = df_temp[feature].skew()
    if messages:
        print(f"Starting skew: {round(skew, 5)}")
    while (round(skew, 2) != 0) and (exp_val <= max_power):
        exp_val += 0.01
        if skew > 0:
            skew = np.power(df_temp[feature], 1/exp_val).skew()
        else:
            skew = np.power(df_temp[feature], exp_val).skew()
    if messages:
        print(f"Final skew: {round(skew, 5)} (using exponent: {round(exp_val, 5)})")

    # Make the transformed version of the feature in the df DataFrame
    if (skew > -0.1) and (skew < 0.1):
        if skew > 0:
            corrected = np.power(df[feature], 1/round(exp_val, 3))
            name = f"{feature}_1/{round(exp_val, 3)}"
        else:
            corrected = np.power(df[feature], round(exp_val, 3))
            name = f"{feature}_{round(exp_val, 3)}"
        # Add the corrected feature to the original DataFrame
        df[name] = corrected
    else:
        name = f"{feature}_binary"
        df[name] = df[feature]
        if skew > 0:
            df.loc[df[name] == df[name].value_counts().index[0], name] = 0
            df.loc[df[name] != df[name].value_counts().index[0], name] = 1
        else:
            df.loc[df[name] == df[name].value_counts().index[0], name] = 1
            df.loc[df[name] != df[name].value_counts().index[0], name] = 0
        if messages:
            print(f"The feature {feature} could not be transformed into a normal distribution.")
            print("Instead, it has been transformed into a binary (0/1) distribution.")

    # Plot visualisations
    if messages:
        fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=FIG_DPI)
        sns.despine(left=True)
        sns.histplot(df_temp[feature], color="b", ax=axs[0], kde=True)
        if (skew > -0.1) and (skew < 0.1):
            if skew > 0:
                corrected = np.power(df_temp[feature], 1/round(exp_val, 3))
            else:
                corrected = np.power(df_temp[feature], round(exp_val, 3))
            df_temp["corrected"] = corrected
            sns.histplot(df_temp["corrected"], color="g", ax=axs[1], kde=True)
        else:
            df_temp["corrected"] = df_temp[feature]
            if skew > 0:
                df_temp.loc[df_temp["corrected"] == df_temp["corrected"].min(), "corrected"] = 0
                df_temp.loc[df_temp["corrected"] > df_temp["corrected"].min(), "corrected"] = 1
            else:
                df_temp.loc[df_temp["corrected"] == df_temp["corrected"].max(), "corrected"] = 1
                df_temp.loc[df_temp["corrected"] < df_temp["corrected"].max(), "corrected"] = 0
            sns.countplot(data=df_temp, x="corrected", color="g", ax=axs[1])
        plt.suptitle(f"Skew of {feature} before and after transformation", fontsize=29)
        plt.setp(axs, yticks=[])
        plt.tight_layout()
        plt.show()
    return df


def test_skew_correct(df):
    for col in df.columns:
        df = skew_correct(df_insurance, col)


## Missing Data
##*********************##

### Missing Not At Random (MNAR)
#### Replace with a theoretical value




### Missing Completely At Random (MCAR)
#### Impute a value




### Missing At Random (MAR)
#### Drop columns/rows or
##### replace with mean/median/mode

def missing_drop(df, label="", features=[], row_threshold=0.90, col_threshold=0.50, messages=True):
    import pandas as pd

    pd.set_option("mode.copy_on_write", True)

    start_count = df.count().sum()
    # Drop all columns that have less data than the proportion col_threshold requires
    col_thresh_val = round((col_threshold * df.shape[0]), 0)
    missing_col_thresh = df.shape[0] - col_thresh_val
    if messages:
        print(start_count, "out of", df.shape[0] * df.shape[1], "in", df.shape[0], "rows(s)")
        print(f"Going to drop any column with more than {missing_col_thresh} missing value(s).")
    df.dropna(axis=1, thresh=col_thresh_val, inplace=True)
    # Drop all rows that have less data than the proportion row_threshold requires
    row_thresh_val = round((row_threshold * df.shape[1]), 0)
    missing_row_thresh = df.shape[1] - row_thresh_val
    if messages:
        print(start_count, "out of", df.shape[0] * df.shape[1], "in", df.shape[1], "column(s)")
        print(f"Going to drop any row with more than {missing_row_thresh} missing value(s).")
    df.dropna(axis=0, thresh=row_thresh_val, inplace=True)
    # Drop all column(s) of given label(s)
    if label != "":
        df.dropna(axis=0, subset=[label], inplace=True)
        if messages:
            print(f"Dropped all column(s) with {label} feature(s).")
    # Function to generate table of residuals if rows/columns with missing values are dropped
    def generate_missing_table():
        df_results = pd.DataFrame(columns=["num_missing", "after_column_drop", "after_rows_drop"])
        for feat in df:
            missing = df[feat].isna().sum()
            if missing > 0:
                rem_col = df.drop(columns=[feat]).count().sum()
                rem_rows = df.dropna(subset=[feat]).count().sum()
                df_results.loc[feat] = [missing, rem_col, rem_rows]
        return df_results
    df_results = generate_missing_table()
    while df_results.shape[0] > 0:
        max_val = df_results[["after_column_drop", "after_rows_drop"]].max(axis=1)[0]
        max_val_axis = df_results.columns[df_results.isin([max_val]).any()][0]
        print(max_val, max_val_axis)
        df_results.sort_values(by=[max_val_axis], ascending=False, inplace=True)
        if messages:
            print("\n", df_results)
        if max_val_axis == "after_rows_drop":
            df.dropna(axis=0, subset=[df_results.index[0]], inplace=True)
        else:
            df.drop(columns=[df_results.index[0]], inplace=True)
        df_results = generate_missing_table()
    if messages:
        print(f"{round(((df.count().sum() / start_count) * 100), 2)}% ({df.count().sum()} out of {start_count}) of non-null cells were kept after dropping.")
    # Return the final DataFrame
    return df



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

parse_column_as_date(df_airbnb, features=["last_review"])
parse_column_as_date(df_airbnb, features=["last_review"], days_to_today=True)


bin_categories(df_airbnb, features=["neighbourhood"], cutoff=0.025)


clean_outlier_per_column(df_insurance, features=df_insurance.columns)
df_insurance.sort_values(by=["bmi"], ascending=False).head()
df_ins_without_outliers = clean_outlier_per_column(df_insurance, features=df_insurance.columns)
df_ins_without_outliers.sort_values(by=["bmi"], ascending=False).head()


df_insurance_ohne_outliers = clean_outlier_by_all_columns(df_insurance)
df_insurance_ohne_outliers.head()
df_insurance_ohne_outliers.sample(RANDOM_SAMPLE_SIZE)

df_nba_salaries_ohne_outliers = clean_outlier_by_all_columns(df_nba_salaries)
df_airbnb_ohne_outliers = clean_outlier_by_all_columns(df_airbnb)
df_airline_satisfaction_ohne_outliers = clean_outlier_by_all_columns(df_airline_satisfaction)


skew_correct(df_insurance, "charges")
skew_correct(df_nba_salaries, "Salary")
skew_correct(df_airbnb, "price")
skew_correct(df_airbnb, "number_of_reviews")
skew_correct(df_airline_satisfaction, "Flight Distance")
skew_correct(df_airline_satisfaction, "Departure Delay in Minutes")
skew_correct(df_airline_satisfaction, "Arrival Delay in Minutes")


test_skew_correct(df_airbnb)
test_skew_correct(df_insurance)


missing_drop(df_insurance.copy())
df_insurance.isna().sum()
missing_drop(df_insurance.copy()).isna().sum()
df_airbnb_ohne_missing = missing_drop(df_airbnb.copy())
df_airbnb.isna().sum()
df_airbnb_ohne_missing.isna().sum()

