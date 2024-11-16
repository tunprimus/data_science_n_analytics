import pandas as pd
import numpy as np
import random
import datetime

pd.set_option("mode.copy_on_write", True)

random.seed(42)

df = pd.DataFrame(
    {
        "date_index": random.sample(range(1, 101), 100),
        "surname": random.choices(["kim", "stone", "alberto", "windsor", "ramirez", "olu"], k=100),
        "firstname": random.choices(["liam", "noel", "paul", "elena", "judith", "enitan"], k=100),
        "middlename_initial": random.choices(["L", "N", "P", "E", "J", "A"], k=100),
        "performance": random.choices(["superb", "good", "okay", "poor", "bad", "woeful"], k=100),
        "salary": random.choices(range(1, 1000), k=100),
        "working_hours": random.choices(range(20, 50), k=100),
    }
)

df.sort_values("date_index", inplace=True)
print(df)
df.to_csv("rolling_features_data.csv", index=False)

df["last_performance"] = df.groupby("surname")["performance"].shift(1)
df["last_1_performance"] = df.groupby("surname")["performance"].shift(2)
df["last_2_performance"] = df.groupby("surname")["performance"].shift(3)

df["moving_working_hours_mean"] = df.groupby("surname")["working_hours"].shift(1).expanding().mean()
df["last_3_working_hours_mean"] = df.groupby("surname")["working_hours"].shift(1).rolling(3).mean()

df["cumulative_salary"] = df.groupby("surname")["salary"].shift(1).expanding().sum()
df["last_3_salary"] = df.groupby("surname")["salary"].shift(1).rolling(3).sum()
print(df)
df.to_csv("rolling_features_result.csv", index=False)
