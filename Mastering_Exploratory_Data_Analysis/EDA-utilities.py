import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import xgboost as xgb
from os.path import realpath as realpath
from scipy.special.agm import agm as agm
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

FIGURE_HEIGHT = 6
GOLDEN_RATIO = 1.618
FIGURE_WIDTH = FIGURE_HEIGHT * GOLDEN_RATIO
FIGURE_DPI = 72
TEST_SIZE = 0.19
RANDOM_SEED = 42

"""
Exploratory Data Analysis (EDA) typically consists of several key components or stages that guide data scientists through the process of understanding and exploring a dataset. These components can vary depending on the specific goals of the analysis and the characteristics of the data, but commonly include:

1) Data collection
2) Data cleaning and preprocessing
3) Descriptive statistics
4) Univariate analysis
5) Bivariate analysis
6) Multivariate analysis
7) Feature engineering
8) Visualisation

EDA_L1: Pure Understanding of Original Data
Basic check on the column datatype, null counts, distinct values and top 10 counts

EDA_L2: Transformation of Original Data
i. Change column names to all be in small letters and spaces to underscore.
ii. Fill in the empty null / NaN rows with reasonable values; after visualisation
iii. Change the datatype of each column to more appropriate ones.
iv. Do data validation 
v. Mapping / binning of categorical features

EDA_L3: Understanding of Transformed Data
i. Correlation analysis
ii. Information value (IV; quantifies the prediction power of a feature) / WOE Values
    - look for IV of 0.1 to 0.5
        < 0.02      ---> useless for prediction
        0.02 - 0.1  ---> weak predictor
        0.1 - 0.3   ---> medium predictor
        0.3 - 0.5   ---> strong predictor
        > 0.5       ---> suspicious or too good to be true
iii. Feature importance from models
iv. Statistical tests
v. Further data analysis on imputed data

"""

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
    Example
    -------
    summary_df = column_summary(df)
    print(summary_df)
    """
    summary_data = []

    df_columns = df.columns
    for col_name in df_columns:
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
    
    Example
    -------
    summary_df = column_summary_plus(df)
    print(summary_df)
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
    Example
    -------
    download_csv_json(df, "/home/some_dir/file_1")
    '''
    dtype_dict = pdf.dtypes.apply(lambda x: str(x)).to_dict()

    with open(json_file_path, "w") as json_file:
        json.dump(dtype_dict, json_file)
    
    return dtype_dict

def download_csv_json(df, main_path):
    '''
    Parameters
    ----------
    df : pandas.DataFrame
        pandas.DataFrame to be saved to csv
    main_path : str
        the path to the csv file to be saved
        
    Returns
    -------
    Tuple
        (csv_path, json_fp)
    
    Save a pandas.DataFrame to csv and json file path.
    The csv file will be saved with the name given in main_path.
    The json file will be saved with the name given in main_path with "_dtype" added to the end.
    The json file will contain the dtype information of the pandas.DataFrame.
    '''
    csv_path = f"{main_path}".csv
    json_fp = f"{main_path}_dtype.json"
    
    dtypedict = dtype_to_json(df, json_fp)
    df.to_csv(csv_path, index=False)

    return csv_path, json_fp


### To Load CSV to Pandas
def json_to_dtype(json_file_path):
    '''
    Parameters
    ----------
    json_file_path : str
        the path to the json file which stores the pandas dtype dictionary

    Returns
    -------
    dict
        the pandas dtype dictionary loaded from the json file
    '''
    with open(json_file_path, "r") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict

def csv_to_pandas(csv_path, json_path):
    '''
    Parameters
    ----------
    csv_path : str
        the path to the csv file which stores the pandas.DataFrame
    json_path : str
        the path to the json file which stores the pandas dtype dictionary
        
    Returns
    -------
    pandas.DataFrame
        the pandas.DataFrame loaded from the csv file with dtype loaded from the json file
    
    Example
    -------
    csvfp = "/home/some_dir/file_1.csv"
    jsonfp = "/home/some_dir/file_1_dtype.json"
    df = csv_to_pandas(csvfp, jsonfp)
    '''
    dtypedict = json_to_dtype(json_path)
    pdf = pd.read_csv(csv_path, dtype=dtypedict)

    return pdf


def dataframe_preview(df):    
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    print(df.head())
    print(df.describe())
    print(df.duplicated().sum())

# Identify numerical columns
def numerical_columns_identifier(df):
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    # Perform univariate analysis on numerical columns
    for column in numerical_columns:
        # For continuous variables
        if len(df[column].unique()) > 10: # assuming if unique values > 10, consider it continuous
            plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
            sns.histplot(df[column], kde=True)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()
        else: # For discrete or ordinal variables
            plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
            ax = sns.countplot(x=column, data=df)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Count")

            # Annotate each bar with its count
            for p in ax.patches:
                ax.annotate(format(p.get_height(), ".0f"), (p.get_x() + p.get_width() / 2., p.get_height()), ha="center", va="center", xytext=(0, 5), textcoords="offset points")
            plt.show()


### Rename the column names for familiarity
# This is if there is no requirement to use back the same column names.
# This is also only done if there is no pre-existing format, or if the col names do not follow conventional format.
# Normally will follow feature mart / dept format to name columns for easy understanding across board.

def rename_columns(df):
    df_l1 = df.copy()
    df_l1.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

    return df_l1


def explore_nulls_nans(df):
    df_l1 = df.copy()
    sns.set(style="whitegrid")

    categorical_columns = df_l1.select_dtypes(exclude=[np.number]).columns

    for col in categorical_columns:
        # Create strip plot
        sns.stripplot(data=df_l1, x=col, y=None, hue=None, order=None, hue_order=None, jitter=True, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor="matplotlib color", linewidth=0, hue_norm=None, log_scale=None, native_scale=False, formatter=None, legend="auto", ax=None)

        # Create violin plot
        sns.violinplot(data=df_l1, x=col, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, fill=True, inner="box", split=False, width=0.8, dodge="auto", gap=0, linewidth=None, linecolor="auto", cut=2, gridsize=100, bw_method="scott", bw_adjust=1, density_norm="area", common_norm=False, hue_norm=None, formatter=None, log_scale=None, native_scale=False, legend="auto", inner_kws=None, ax=None)

        # Create boxplot
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        sns.boxplot(x=col, y="y", data=df_l1)
        # Set labels and title
        plt.xlabel(f"{col}")
        plt.ylabel("{y}")
        plt.title(f"Boxplot of y by {col}")
        plt.yscale("log")
        # Show the plot
        plt.xticks(rotation=45)  # rotate x-axis labels for better readability
        plt.tight_layout()  # adjust layout to prevent clipping of labels
        plt.show()


def selective_fill_nans(df):
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    try:
        for i in numerical_columns[df.isnull().any(axis=0)]:
            df[i].fillna(df[i].agm(), inplace=True)
    except ValueError:
        for i in df.columns[df.isnull().any(axis=0)]:
            df[i].fillna(df[i].agm(), inplace=True)


def explore_correlation(df):
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    correlation_matrix = df[numerical_columns].corr()

    # Create the heatmap
    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    # Set title
    plt.title("Correlation Heatmap")
    # Show the plot
    plt.tight_layout()
    plt.show()

    # Find the max correlation
    upper_triangular = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    max_correlation = upper_triangular.max().max()
    print(f"Maximum pairwise correlation: {max_correlation:.2f}")

def display_pairwise_correlation(df_input, col_1, col_2):
    numerical_columns = df_input.select_dtypes(include=[np.number]).columns

    for index, _ in enumerate(numerical_columns):
        col_1 = numerical_columns[index]
        col_2 = numerical_columns[index - 1] if index > 1 else None

        correlation_value = df_input[col_1].corr(df_input[col_2])
        return f"Correlation value between {col_1} and {col_2} is: {correlation_value}"


def iv_woe(data, target, bins=10, show_woe=False):
    # Create empty DataFrames
    new_df, woe_df = pd.DataFrame(), pd.DataFrame()

    # Extract column names
    cols = data.columns

    # Run WOE and IV on all the independent variables
    for i_vars in cols[~cols.isin([target])]:
        print("Processing variable:", i_vars)
        if (data[i_vars].dtype.kind in "bifc") and (len(np.unique(data[i_vars])) > 10):
            binned_x = pd.qcut(data[i_vars], bins, duplicates="drop")
            buffer_df = pd.DataFrame({"x": binned_x, "y": data[target]})
        else:
            buffer_df = pd.DataFrame({"x": data[i_vars], "y": data[target]})
        
        # Calculate the number of events in each group (bin)
        evt = buffer_df.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        evt.columns = ["Cutoff", "N", "Events"]

        # Calculate the % of events in each group
        evt["%_of_Events"] = np.maximum(evt["Events"], 0.5) / evt["Events"].sum()

        # Calculate the non-events in each group
        evt["Non-Events"] = evt["N"] - evt["Events"]
        # Calculate the % of non-events in each group
        evt["%_of_Non-Events"] = np.maximum(evt["Non-Events"], 0.5) / evt["Non-Events"].sum()

        # Calculate WOE by taking natural log of division of % of non-events and % of events
        evt["WoE"] = np.log(evt["%_of_Events"] / evt["%_of_Non-Events"])
        evt["IV"] = evt["WoE"] * (evt["%_of_Events"] - evt["%_of_Non-Events"])
        evt.insert(loc=0, column="Variable", value=i_vars)
        print("Information value of " + i_vars + " is " + str(round(evt["IV"].sum(), 6)))
        temp = pd.DataFrame({"Variable": [i_vars], "IV": [evt["IV"].sum()]}, columns=["Variable", "IV"])
        new_df = pd.concat([new_df, temp], axis=0)
        woe_df = pd.concat([woe_df, evt], axis=0)

        # Shoe WOE table
        if show_woe is True:
            print(evt)
    return new_df, woe_df


def column_categoriser(df, all_col=False):
    buffer_df = df.copy()
    numerical_columns = buffer_df.select_dtypes(include=[np.number]).columns
    categorical_columns = buffer_df.select_dtypes(exclude=[np.number]).columns
    dependent_column = ["target_encoded"]
    independent_column = numerical_columns + categorical_columns
    all_columns = numerical_columns + categorical_columns + dependent_column

    if all_col is True:
        return numerical_columns, independent_column, dependent_column, all_columns
    else:
        return numerical_columns, independent_column, dependent_column


def model_data_preprocessor(df):
    buffer_df = df.copy()
    numerical_cols, independent_col, dependent_col = column_categoriser(buffer_df, all_col=False)

    X_train, X_test, y_train, y_test = train_test_split(buffer_df[independent_col], buffer_df[dependent_col],\
                                                    stratify=buffer_df[dependent_col], test_size=TEST_SIZE, random_state=RANDOM_SEED)
    preprocessor = ColumnTransformer(\
        transformers=[("num", StandardScaler(), numerical_cols)]
        )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=independent_col)
    X_test_transformed = preprocessor.fit_transform(X_test)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=independent_col)
    y_train_transformed = y_train.values.ravel()
    y_test_transformed = y_test.values.ravel()

    return (X_train, X_test, y_train, y_test, X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed, X_train_transformed_df, X_test_transformed_df)




