#!/usr/bin/env python3
import json


with open("nested_people.json", mode="r", encoding="utf-8") as read_file:
    people = json.load(read_file)
    print(people)

    sort_by_age = dict(sorted(people.items(), key=lambda x: x[1]["age"], reverse=True))
    print(sort_by_age)

    sort_by_key = dict(sorted(people.items(), key=lambda x: x[0], reverse=True))
    print(sort_by_key)
