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

def generate_category(person):
    if ((person["working_hours"] > 40) | (person["salary"] > 500)):
        return "A"
    else:
        return "B"

df["category"] = df.apply(generate_category, axis=1)
print(df)
df.to_csv("generated_category_result.csv", index=False)
