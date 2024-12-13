import pandas as pd
import numpy as np

pd.set_option("mode.copy_on_write", True)

df = pd.DataFrame({
    "A": [1, None, 3, 4, 5],
    "B": [5, None, None, 4, 5],
    "C": [None, 2, 3, 4, 5],
    "D": [11, None, 13, 14, 15],
    "E": [5, 7, 13, 8, None],
    "F": [12, 22, None, 34, 45],
    "G": [9, 14, 3, 8, 5],
    "H": [None, 4, 3, 4, 5],
    "I": [11, 17, 13, None, 25],
    "J": [21, None, 33, 44, 55]
})

# Method 1: Using DataFrame.fillna() with DataFrame.median()
df01 = df.copy(deep=True)
df01.fillna(df01.median(), inplace=True)
print(df01)

# Method 2: Using DataFrame.apply() for Selective Column Filling
df02 = df.copy(deep=True)
df02 = df02.apply(lambda col: col.fillna(col.median()))
print(df02)

# Method 3: Filling with Median using DataFrame.transform()
df03 = df.copy(deep=True)
df03 = df03.transform(lambda col: col.fillna(col.median()))
print(df03)

# Method 4: Using numpy.where() with DataFrame.isnull()
df04 = df.copy(deep=True)
for column in df04.columns:
    df04[column] = np.where(df04[column].isnull(), df04[column].median(), df04[column])

print(df04)

# Method 5: Using DataFrame.where() with notnull()
df05 = df.copy(deep=True)
df05 = df05.where(df05.notnull(), df05.median(), axis="columns")
print(df05)

