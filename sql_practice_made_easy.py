import sqlite3
import pandas as pd
from os.path import realpath as realpath

pd.set_option("mode.copy_on_write", True)

# Connect to the Chinook SQLite database
conn = sqlite3.connect(realpath("../.assets/data/chinook.db"))

# Query to view all tables
tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(tables_query, conn)
print(tables)

# Query the customers table
customers_query = "SELECT * FROM customers LIMIT 11;"
customers = pd.read_sql(customers_query, conn)
print(customers)

# Join the Track and Album tables to get track names and album titles
track_album_join_query = """
SELECT tracks.Name AS tracks_name, albums.Title AS albums_title
FROM tracks
JOIN albums ON tracks.AlbumId = albums.AlbumId
LIMIT 11;
"""
track_album = pd.read_sql(track_album_join_query, conn)
print(track_album)
