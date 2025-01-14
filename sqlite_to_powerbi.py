#!/usr/bin/env python3
import sqlite3
import pandas as pd
from os.path import realpath as realpath

pd.set_option("mode.copy_on_write", True)

# Connect to the SQLite database
conn = sqlite3.connect(realpath("../.assets/data/chinook.db"))

# Function to load all tables into a dictionary of DataFrames
def load_all_tables(conn):
    tables = {}
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()

    for table_name in table_names:
        table_name = table_name[0]
        tables[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    return tables

# Load all tables into a dictionary of DataFrames
tables = load_all_tables(conn)

# Display the DataFrame of a specific table, e.g., "album"
print(tables["albums"].head())

# Loop over all tables and print them
for idx, table_name in enumerate(tables):
    print(table_name)
    print(tables[f"{table_name}"].head())

# Close the connection
conn.close()
