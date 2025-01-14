#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random
import datetime

pd.set_option("mode.copy_on_write", True)

random.seed(42)

df = pd.DataFrame(
    {
        "surname": random.choices(["kim", "stone", "alberto", "windsor"], k=100),
        "firstname": random.choices(["liam", "noel", "justus", "elena"], k=100),
        "city": random.choices(["paris", "london", "istanbul", "seoul"], k=100),
        "id": random.choices(range(1, 4), k=100),
        "salary": random.choices(range(1, 1000), k=100),
        "working_hours": random.choices(range(20, 50), k=100),
        "date": random.choices(
            [
                datetime.date(2022, 1, 22),
                datetime.date(2021, 1, 22),
                datetime.date(2020, 1, 22),
                datetime.date(2019, 1, 22),
                datetime.date(2018, 1, 22),
                datetime.date(2017, 1, 22),
                datetime.date(2016, 1, 22),
                datetime.date(2015, 1, 22),
                datetime.date(2014, 1, 22),
                datetime.date(2013, 1, 22),
                datetime.date(2012, 1, 22),
            ],
            k=100,
        ),
    }
)
print(df)
df.to_csv("groupby_work_data.csv", index=False)

result = df.groupby(["surname", "city", "id"]).agg(
    min_date=("date", "min"),
    max_date=("date", "max"),
    min_salary=("salary", "min"),
    max_salary=("salary", "max"),
    mean_salary=("salary", "mean"),
    median_salary=("salary", "median"),
    min_working_hours=("working_hours", "min"),
    max_working_hours=("working_hours", "max"),
    mean_working_hours=("working_hours", "mean"),
    median_working_hours=("working_hours", "median"),
).reset_index()
print(result)
result.to_csv("groupby_result.csv", index=False)
