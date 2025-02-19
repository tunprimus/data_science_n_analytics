#!/usr/bin/env python3
import sqlite3
from os.path import realpath as realpath

import numpy as np
import pandas as pd

# Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

RANDOM_SAMPLE_SIZE = 13
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72
PETAJOULES_TO_GIGAJOULES_VAL = 1000000

# Path to Datasets
path_to_energy_data = "../000_common_dataset/from-AltSchoolAfrica-Energy_Indicators.xls"
path_to_gdp_data = "../000_common_dataset/from-AltSchoolAfrica-world_bank.csv"
path_to_journal_data = "../000_common_dataset/from-AltSchoolAfrica-scimagojr-3.xlsx"

real_path_to_energy_data = realpath(path_to_energy_data)
real_path_to_gdp_data = realpath(path_to_gdp_data)
real_path_to_journal_data = realpath(path_to_journal_data)


# Function to view preliminary info from DataFrame
def df_preliminary_info(df):
    print("------ DataFrame Random Sample: -----\n")
    print(df.sample(RANDOM_SAMPLE_SIZE))
    print("\n------ DataFrame Info: -----\n")
    print(df.info())
    print("\n------ DataFrame Head: -----\n")
    print(df.head())
    print("\n------ DataFrame Tail: -----\n")
    print(df.tail())
    print("\n------ DataFrame Describe: -----\n")
    print(df.describe())


# ======================================= #
# Load Energy Dataset with Transformation
# ======================================= #
def generate_energy_dataframe(real_path_to_energy_data, header_to_skip=16, footer_to_skip=38, rows_to_skip=1, cols_to_use=[2, 3, 4, 5], names_to_use=["country", "energy_supply", "energy_supply_per_capita", "pct_renewable"], nan_value="..."):
    df = pd.read_excel(
        real_path_to_energy_data,
        header=header_to_skip,
        skipfooter=footer_to_skip,
        skiprows=rows_to_skip,
        usecols=cols_to_use,
        names=names_to_use,
        na_values=nan_value,
    )
    # Convert energy supply to gigajoules
    df["energy_supply"] *= PETAJOULES_TO_GIGAJOULES_VAL
    # Remove text with parenthesis, including the parentheses themselves
    df["country"] = df["country"].str.replace(r"\s*\([^)]*\)", "", regex=True)
    # Remove the numbers
    df["country"] = df["country"].str.replace(r"\d+", "", regex=True)
    # Rename some countries via dictionary
    rename_countries01 = {
        "Republic of Korea": "South Korea",
        "Democratic People's Republic of Korea": "North Korea",
        "Lao People's Democratic Republic": "Lao PDR",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "China, Hong Kong Special Administrative Region": "Hong Kong",
        "China, Macao Special Administrative Region": "Macao SAR, China",
        "The former Yugoslav Republic of Macedonia": "Macedonia, FYR",
        "United States Virgin Islands": "Virgin Islands (U.S.)",
        "Viet Nam": "Vietnam",
        "Democratic Republic of the Congo": "DR Congo",
        "United Republic of Tanzania": "Tanzania",
        "Bonaire, Sint Eustatius and Saba": "Bonaire, Saba and Sint Eustatius",
        "Congo": "Republic of Congo",
        "Faeroe Islands": "Faroe Islands",
        "Sint Maarten": "Sint Maarten (Netherlands)",
        "British Virgin Islands": "Virgin Islands (U.K.)",
        "State of Palestine": "West Bank and Gaza",
    }
    df["country"] = df["country"].replace(rename_countries01)
    for i in df["country"]:
        print(i)
    return df

Energy = generate_energy_dataframe(real_path_to_energy_data, header_to_skip=16, footer_to_skip=38, rows_to_skip=1, cols_to_use=[2, 3, 4, 5], names_to_use=["country", "energy_supply", "energy_supply_per_capita", "pct_renewable"], nan_value="...")
df_preliminary_info(Energy)


# ======================================= #
# Load GDP Dataset with Transformation
# ======================================= #
def generate_gdp_dataframe(real_path_to_gdp_data, num_rows_to_skip=4):
    df = pd.read_csv(real_path_to_gdp_data, skiprows=num_rows_to_skip)
    df.rename(
        columns=lambda x: x.lower()
        .replace(" ", "_")
        .replace("country_name", "country"),
        inplace=True,
    )
    # Rename some countries via dictionary
    rename_countries02 = {
        "Korea, Rep.": "South Korea",
        "Iran, Islamic Rep.": "Iran",
        "Hong Kong SAR, China": "Hong Kong",
        "Bahamas, The": "Bahamas",
        "Egypt, Arab Rep.": "Egypt",
        "Micronesia, Fed. Sts.": "Micronesia",
        "Kyrgyz Republic": "Kyrgyzstan",
        "St. Kitts and Nevis": "Saint Kitts and Nevis",
        "St. Martin (French part)": "Saint Martin",
        "Korea, Dem. People’s Rep.": "North Korea",
        "Slovak Republic": "Slovakia",
        "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
        "Venezuela, RB": "Venezuela",
        "British Virgin Islands": "Virgin Islands (U.K.)",
        "Yemen, Rep.": "Yemen",
        "Congo, Dem. Rep.": "DR Congo",
        "United States": "United States of America",
        "St. Lucia": "Saint Lucia",
        "Gambia, The": "Gambia",
        "Congo, Rep.": "Republic of Congo",
    }
    df["country"] = df["country"].replace(rename_countries02)
    for i in df["country"]:
        print(i)
    return df

GDP = generate_gdp_dataframe(real_path_to_gdp_data, num_rows_to_skip=4)
df_preliminary_info(GDP)


# ======================================= #
# Load Journal Dataset with Transformation
# ======================================= #
def generate_journal_dataframe(real_path_to_journal_data):
    df = pd.read_excel(real_path_to_journal_data)
    df.rename(
        columns=lambda x: x.lower()
        .replace(" ", "_"),
        inplace=True,
    )
    # Rename some countries via dictionary
    rename_countries03 = {
        "United States": "United States of America",
        "Viet Nam": "Vietnam",
        "Congo": "Republic of Congo",
        "Syrian Arab Republic": "Syria",
        "Palestine": "West Bank and Gaza",
        "Laos": "Lao PDR",
        "Macedonia": "Macedonia, FYR",
    }
    df["country"] = df["country"].replace(rename_countries03)
    for i in df["country"]:
        print(i)
    return df


ScimEn = generate_journal_dataframe(real_path_to_journal_data)
df_preliminary_info(ScimEn)

