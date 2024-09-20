import numpy as np
import pandas as pd
import json
from os.path import realpath as realpath

# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)


def column_summary(df):
    """
    Creates a summary of a given DataFrame.

    For each column in the DataFrame, it returns the following information:
        - The column's data type
        - The number of null values
        - The number of non-null values
        - The number of distinct values
        - A dictionary where the keys are the distinct values and the values are the counts of each distinct value. If the number of distinct values is larger than 10, it will only return the top 10 most frequent values.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be summarised.

    Returns
    -------
    DataFrame
        A DataFrame containing the summary of the given DataFrame.
    """
    summary_data = []

    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_of_nulls = df[col_name].isnull().sum()
        num_of_nun_nulls = df[col_name].notnull.sum()
        num_of_distinct_values = df[col_name].nunique()

        if num_of_distinct_values <= 10:
            distinct_values_counts = df[col_name].value_counts().to_dict()
        else:
            top_10_values_counts = df[col_name].value_counts().head(10).to_dict()
            distinct_values_counts = {
                k: v
                for k, v in sorted(
                    top_10_values_counts.items(), key=lambda item: item[1], reverse=True
                )
            }

        summary_data.append(
            {
                "col_name": col_name,
                "col_dtype": col_dtype,
                "num_of_nulls": num_of_nulls,
                "num_of_nun_nulls": num_of_nun_nulls,
                "num_of_distinct_values": num_of_distinct_values,
                "distinct_values_counts": distinct_values_counts,
            }
        )

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def column_summary_plus(df):
    """
    Creates a summary of a given DataFrame, including the following information:
        - The column's data type
        - The number of distinct values
        - The minimum and maximum values
        - The median value of non-null values
        - The average value of non-null values
        - The average value of non-zero values
        - Whether null values are present
        - The number of null values
        - The number of non-null values
        - A dictionary where the keys are the top 10 distinct values and the values are the counts of each distinct value.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be summarised.

    Returns
    -------
    DataFrame
        A DataFrame containing the summary of the given DataFrame.
    """
    result_df = pd.DataFrame(columns=["col_name", "col_dtype", "num_of_distinct_values", "min_value", "max_value", "median_no_na", "average_no_na", "average_non_zero", "null_present", "nulls_num", "non_nulls_num",  "distinct_values"])

    df_columns = df.columns
    # Loop through each column in the DataFrame
    for column in df_columns:
        print(f"Start processing {column} col with {df[column].dtype} dtype")
        # Get column dtype
        col_dtype = df[column].dtype
        # Get distinct values and their counts
        value_counts = df[column].value_counts()
        distinct_values = value_counts.index.tolist()
        # Get number of distinct values
        num_distinct_values = len(distinct_values)
        # Get min and max values
        sorted_values = sorted(distinct_values)
        min_value = sorted_values[0] if sorted_values else None
        max_value = sorted_values[-1] if sorted_values else None

        # Get median value
        non_distinct_val_list = sorted(df[column].dropna().tolist())
        len_non_d_list = len(non_distinct_val_list)
        if len(non_distinct_val_list) == 0:
            median = None
        else:
            median = non_distinct_val_list[len_non_d_list // 2]
        
        # Get average value if value is number
        if np.issubdtype(df[column].dtype, np.number):
            if len(non_distinct_val_list) > 0:
                average = sum(non_distinct_val_list) / len_non_d_list
                non_zero_val_list = [v for v in non_distinct_val_list if v > 0]
                average_non_zero = sum(non_zero_val_list)/len_non_d_list
            else:
                average = None
                average_non_zero = None
        else:
            average = None
            average_non_zero = None

        # Check if null values are present
        null_present = 1 if df[column].isnull().any() else 0

        # Get number of nulls and non-nulls
        num_nulls = df[column].isnull().sum()
        num_non_nulls = df[column].notnull().sum()

        # Distinct_values only take top 10 distinct values count
        top_10_d_v = value_counts.head(10).index.tolist()
        top_10_c = value_counts.head(10).tolist()
        top_10_d_v_dict = dict(zip(top_10_d_v, top_10_c))

        # Append the information to the result DataFrame
        result_df = result_df.append({"col_name": column, "col_dtype": col_dtype, "num_distinct_values": num_distinct_values, "min_value": min_value, "max_value": max_value, "median_no_na": median, "average_no_na": average, "average_non_zero": average_non_zero, "null_present": null_present, "nulls_num": num_nulls, "non_nulls_num": num_non_nulls, "distinct_values": top_10_d_v_dict})
    
    return result_df


### To Save Pandas to CSV
def dtype_to_json(pdf, json_file_path):
    '''
    Parameters
    ----------
    pdf : pandas.DataFrame
        pandas.DataFrame so we can extract the dtype
    json_file_path : str
        the json file path location
        
    Returns
    -------
    Dict
        The dtype dictionary used
    
    To create a json file which stores the pandas dtype dictionary for
    use when converting back from csv to pandas.DataFrame.
    '''
    dtype_dict = pdf.dtypes.apply(lambda x: str(x)).to_dict()

    with open(json_file_path, "w") as json_file:
        json.dump(dtype_dict, json_file)
    
    return dtype_dict

def download_csv_json(df, main_path):
    csv_path = f"{main_path}".csv
    json_fp = f"{main_path}_dtype.json"
    
    dtypedict = dtype_to_json(df, json_fp)
    df.to_csv(csv_path, index=False)

    return csv_path, json_fp


### To Load CSV to Pandas
def json_to_dtype(json_file_path):
    with open(json_file_path, "r") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict

def csv_to_pandas(csv_path, json_path):
    dtypedict = json_to_dtype(json_path)
    pdf = pd.read_csv(csv_path, dtype=dtypedict)

    return pdf

