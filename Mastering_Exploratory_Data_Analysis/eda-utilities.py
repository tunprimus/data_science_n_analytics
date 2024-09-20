import numpy as np
import pandas as pd
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
