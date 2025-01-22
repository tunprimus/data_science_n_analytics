#!/usr/bin/env python3
# https://andrewpwheeler.com/2021/09/14/extending-sklearns-ordinalencoder/
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

class SimpleOrdEnc():
    def __init__(self, dtype=int, unknown_value=-1, rare_cat_keep=None, case_thresh_count=None):
        """
        Initialise the SimpleOrdEnc class with optional parameters.

        Parameters:
        dtype : type, default=int
            The data type to which the categories should be converted.
        unknown_value : int or None, default=-1
            The value to use for unknown categories during transformation.
        rare_cat_keep : any, default=None
            The value to assign to rare categories, if specified.
        case_thresh_count : int or None, default=None
            The threshold count to determine rare categories, if specified.
        """
        self.dtype = dtype
        self.unknown_value = unknown_value
        self.rare_cat_keep = rare_cat_keep
        self.case_thresh_count = case_thresh_count
        self.vars = None
        self.soe = None
    # Method to fit the data
    def fit(self, X):
        """
        Fit the SimpleOrdEnc object to the data.

        Parameters:
        X : pd.DataFrame
            The dataframe to which the OrdinalEncoder should be fit.

        Returns:
        self : SimpleOrdEnc
            The fit object.
        """
        self.vars = list(X)
        # now creating fit for each variable
        res_oe = {}
        for val in list(X):
            res_oe[val] = OrdinalEncoder(dtype=self.dtype, handle_unknown="use_encoded_value", unknown_value=self.unknown_value)
            # get unique values minus missing
            x_counts = X[val].value_counts().reset_index()
            # if rare_cat_keep, only take top K value
            if self.rare_cat_keep:
                top_k = self.rare_cat_keep - 1
                uniq_vals = x_counts.loc[0:top_k, :]
            # if count, use that to filter
            elif self.case_thresh_count:
                uniq_vals = x_counts[x_counts[val] >= self.case_thresh_count].copy()
            # if neither
            else:
                uniq_vals = x_counts
            # now fit the ordinal encoder for one variable
            res_oe[val].fit(uniq_vals[["index"]])
        # Append back to the big class
        self.soe = res_oe
    # Method to transform the data
    def transform(self, X):
        """
        Transform the data using the OrdinalEncoder.

        Parameters:
        X : pd.DataFrame
            The dataframe to be transformed.

        Returns:
        pd.DataFrame
            The transformed dataframe.
        """
        x_copy = X[self.vars].copy()
        for val in self.vars:
            x_copy[val] = self.soe[val].transform(X[[val]].fillna(self.unknown_value))
        return x_copy
    # Method to inverse transform the data
    def inverse_transform(self, X):
        """
        Inverse transform the data using the fitted OrdinalEncoder.

        Parameters:
        X : pd.DataFrame
            The dataframe to be inverse transformed.

        Returns:
        pd.DataFrame
            The inverse transformed dataframe with original categorical values.
        """
        x_copy = X[self.vars].copy()
        for val in self.vars:
            x_copy[val] = self.soe[val].inverse_transform(X[[val]].fillna(self.unknown_value))
        return x_copy


# Tests
test_x1 = [1, 2, 3, 4]
test_x2 = ["a", "b", "c", "d"]
test_x3 = ["z", "z", None, "y"]
test_x4 = [4, np.nan, 5, 6]
test_01 = pd.DataFrame(zip(test_x1, test_x2, test_x3, test_x4), columns=["test_x1", "test_x2", "test_x3", "test_x4"])
print(test_01)

ord_encoder = SimpleOrdEnc()
ord_encoder.fit(test_01)
tx_01 = ord_encoder.transform(test_01)
print(tx_01)
invx_01 = ord_encoder.inverse_transform(test_01)
print(invx_01)
