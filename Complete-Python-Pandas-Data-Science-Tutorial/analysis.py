#!/usr/bin/env python3
import pandas as pd

poke = pd.read_csv('pokemon_data.csv')

print(poke.tail(5))

