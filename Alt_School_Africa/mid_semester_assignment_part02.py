#!/usr/bin/env python3
import inspect
import numpy as np
import pandas as pd
import sqlite3
from os.path import realpath as realpath
from ydata_profiling import ProfileReport

# Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

RANDOM_SAMPLE_SIZE = 13
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72
PETAJOULES_TO_GIGAJOULES_VAL = 1_000_000

# Path to Database
path_to_database = "./mid_semester_assignment.db"

real_path_to_database = realpath(path_to_database)


# ^^^^^^^^^^^^^^^^^^^^^^ #
#  Utility Functions
# ^^^^^^^^^^^^^^^^^^^^^^ #

# Function to view preliminary info from DataFrame
def df_preliminary_info(df):
    print("------ DataFrame Random Sample: -----\n")
    print(df.sample(RANDOM_SAMPLE_SIZE))
    print("\n------ DataFrame Info: -----\n")
    print(df.info())
    print("\n------ DataFrame Head: -----\n")
    print(df.head())
    print("\n------ DataFrame Tail: -----\n")
    print(df.tail())
    print("\n------ DataFrame Describe: -----\n")
    print(df.describe())

# Function to store DataFrame inside sqlite database
def save_dataframes_to_sqlite(path_to_db, *dataframes):
    """
    Save multiple DataFrames into different tables inside a single SQLite database.

    Args:
        path_to_db (str): Name of the SQLite database file.
        *dataframes (DataFrames): Variable number of DataFrames to be saved.
    """
    if path_to_db:
        real_path_to_db = realpath(path_to_db)
    else:
        real_path_to_db = realpath(".")
    with sqlite3.connect(real_path_to_db) as conn:
        frame = inspect.currentframe().f_back
        for df in dataframes:
            table_name = [var for var, val in frame.f_locals.items() if val is df][0]
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    print(f"DataFrames saved to {real_path_to_db}")


def save_dataframes_to_sqlite_by_dict_(path_to_db, **dataframes):
    """
    Save multiple DataFrames into different tables inside a single SQLite database.

    Args:
        path_to_db (str): Name of the SQLite database file.
        **dataframes (keyword arguments): DataFrame variables and their corresponding table names.
            Example: df1=pd.DataFrame(), df2=pd.DataFrame()
    """
    if path_to_db:
        real_path_to_db = realpath(path_to_db)
    else:
        real_path_to_db = realpath(".")
    with sqlite3.connect(real_path_to_db) as conn:
        for table_name, df in dataframes.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    print(f"DataFrames saved to {real_path_to_db}")


def save_dataframes_to_sqlite_by_tuples(path_to_db, *dataframes):
    """
    Save multiple DataFrames into different tables inside a single SQLite database.

    Args:
        path_to_db (str): Name of the SQLite database file.
        *dataframes (tuple of tuples): Each tuple contains a DataFrame and its corresponding table name.
            Example: (df1, "table1"), (df2, "table2"), ...
    """
    if path_to_db:
        real_path_to_db = realpath(path_to_db)
    else:
        real_path_to_db = realpath(".")
    with sqlite3.connect(real_path_to_db) as conn:
        for df, table_name in dataframes:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    print(f"DataFrames saved to {real_path_to_db}")


# Parsing a sqlite database into a dictionary of DataFrames without knowing the table names
def fetch_all_tables_from_sqlite(path_to_db):
    """
    Retrieve the contents of a SQLite database and return them as a
    dictionary of DataFrames.

    Parameters
    ----------
    path_to_db : str
        The name of the SQLite database file to read from.

    Returns
    -------
    dict
        A dictionary where each key is the name of a table in the
        database and the value is the contents of that table as a
        DataFrame.
    """
    sql_query = "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%';"
    if path_to_db:
        real_path_to_db = realpath(path_to_db)

    with sqlite3.connect(real_path_to_db) as conn:
        cur = conn.cursor()
        cur.execute(sql_query)
        tables = cur.fetchall()
        db_tables = [table[0] for table in tables]
        out_dict = {tbl: pd.read_sql_query(f"SELECT * FROM {tbl}", conn) for tbl in db_tables}
    return out_dict


def display_sqlite_table_info(path_to_db):
    sql_query = "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%';"
    if path_to_db:
        real_path_to_db = realpath(path_to_db)
    with sqlite3.connect(real_path_to_db) as conn:
        cur = conn.cursor()
        cur.execute(sql_query)
        tables = cur.fetchall()
        db_tables = [table[0] for table in tables]
        for table_name in db_tables:
            print(f"Table: {table_name}")
            cur.execute(f"PRAGMA table_info('{table_name}');")
            columns = cur.fetchall()
            for column in columns:
                print(f"Column: {column[1]}, Type: {column[2]}")
            print("\n")


# ^^^^^^^^^^^^^^^^^^^^^^ #
#  End utility functions
# $$$$$$$$$$$$$$$$$$$$$$ #


# ======================================= #
# Retrieve Energy, GDP and Journal DataFrames from Sqlite
# ======================================= #
display_sqlite_table_info(real_path_to_database)
all_dataframes = fetch_all_tables_from_sqlite(real_path_to_database)

print(all_dataframes.keys())

profile_energy = ProfileReport(all_dataframes["Energy"], title="Profile Report for Energy")
# profile_energy.to_notebook_iframe()

profile_gdp = ProfileReport(all_dataframes["GDP"], title="Profile Report for GDP")
# profile_gdp.to_notebook_iframe()

profile_scim_en = ProfileReport(all_dataframes["ScimEn"], title="Profile Report for ScimEn")
# profile_scim_en.to_notebook_iframe()


