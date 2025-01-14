#!/usr/bin/env python3
import json


with open("nested_people.json", mode="r", encoding="utf-8") as read_file:
    people = json.load(read_file)
    print(people)

    filter_by_age = dict(filter(lambda x: x[1]["age"] > 20, people.items()))
    print(filter_by_age)

    filter_by_key_length = dict(filter(lambda x: len(x[0]) == 5, people.items()))
    print(filter_by_key_length)
