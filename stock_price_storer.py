#!/usr/bin/env python3
import sqlite3
import pandas as pd
from os.path import realpath as realpath


def store_into_database(msg, database_name):
    """
    This function stores data into a SQLite database.

    Args:
    msg (dict): A dictionary where each key is a table name and
                each value is a pandas DataFrame to be stored in the database.

    The function iterates over the dictionary, printing the name of each DataFrame
    and its contents, then storing each DataFrame into a SQLite table with the
    same name. If the table already exists, it is replaced.
    """
    # Connect to SQLite database (or create it if it doesn't exist)
    real_path_to_db = realpath(database_name)
    conn = sqlite3.connect(real_path_to_db)

    # Iterate over the dictionary items
    for name, df in msg.items():
        print(name)  # Print the name of the DataFrame
        print(df)    # Print the contents of the DataFrame
        # Store the DataFrame into a SQL table with the same name
        df.to_sql(name, conn, if_exists="replace", index=True)

    # Commit (save) the changes to the database
    conn.commit()


def get_from_database(database_name):
    """
    This function retrieves all tables from a SQLite database and
    returns them as a dictionary of pandas DataFrames.

    Returns:
    dict: A dictionary where each key is a table name and each value is
          a pandas DataFrame containing the data from the corresponding SQL table.
    """
    # Connect to SQLite database
    real_path_to_db = realpath(database_name)
    conn = sqlite3.connect(real_path_to_db)
    cur = conn.cursor()

    # Retrieve the names of all tables in the database
    cur.execute('SELECT name from sqlite_master where type= "table"')
    data = cur.fetchall()  # Fetch all the results

    option_price_df = {}
    # Iterate over each table name
    for i in data:
        k = i[0]  # Table name
        # Query the table and store the result into a DataFrame
        option_price_df[k] = pd.read_sql_query(f"SELECT * FROM {k}", conn)

    # Return the dictionary of DataFrames
    return option_price_df

