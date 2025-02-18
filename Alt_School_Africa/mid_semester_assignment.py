#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sqlite3
from os.path import realpath as realpath
# Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
np.float = np.float64   
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

# Path to Datasets
path_to_energy_data = "../000_common_dataset/from-AltSchoolAfrica-Energy_Indicators.xls"
path_to_gdp_data = "../000_common_dataset/from-AltSchoolAfrica-world_bank.csv"
path_to_journal_data = "../000_common_dataset/from-AltSchoolAfrica-scimagojr-3.xlsx"


