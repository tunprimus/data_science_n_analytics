#!/usr/bin/env python3
import sqlite3
import pandas as pd
from os.path import realpath as realpath


def store_into_database(msg, database_name):
    """
    Store the contents of the given dictionary into a SQLite database.

    Parameters
    ----------
    msg : dict
        A dictionary where the keys are the names of the DataFrames and
        the values are the DataFrames themselves.

    database_name : str
        The name of the SQLite database file to store the data in.
    """
    # Connect to SQLite database (or create it if it doesn't exist)
    real_path_to_db = realpath(database_name)
    with sqlite3.connect(real_path_to_db) as conn:
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
    Retrieve the contents of a SQLite database and return them as a
    dictionary of DataFrames.

    Parameters
    ----------
    database_name : str
        The name of the SQLite database file to read from.

    Returns
    -------
    dict
        A dictionary where each key is the name of a table in the
        database and the value is the contents of that table as a
        DataFrame.
    """
    # Connect to SQLite database
    real_path_to_db = realpath(database_name)
    with sqlite3.connect(real_path_to_db) as conn:
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

