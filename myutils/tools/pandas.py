# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import math
import random

import numpy as np
import pandas as pd
import bisect

from .shuffle import random_shuffle_with_seed


# %%
class DF_spliter:


    @classmethod
    def split_by_number_and_column(cls, data, splits, refer=None):
        if splits[0] < 1 and sum(splits) == 1.0:
            splits = [s * len(data) for s in splits]
        
        unique_ids = list(data[refer].unique())
        id2num = data[refer].value_counts()
        unique_ids = random_shuffle_with_seed(unique_ids, 0.99)
        numbers = [id2num[x] for x in unique_ids]
        numbers = np.cumsum(numbers)
        
        
        sub_datas = []
        splits = np.cumsum([0] + splits)
        for i in range(1, len(splits)):
            pos_s = bisect.bisect(numbers, splits[i-1])
            pos_e = bisect.bisect(numbers, splits[i])
            _data =  data[data[refer].isin(unique_ids[pos_s:pos_e])].reset_index(drop=True)
            sub_datas.append(_data)
        return sub_datas

        
    @classmethod
    def split_by_number(cls, data, splits):
        # print(len(data), splits, sum(splits))
        assert sum(splits) == len(data) and all([s >= 1 for s in splits])
        N = len(data)
        N_index = list(range(N))
        random.seed(42)
        random.shuffle(N_index)

        splits = np.cumsum([0] + splits)
        sub_datas = []
        for i in range(1, len(splits)):
            _data = data.iloc[N_index[splits[i - 1] : splits[i]]].reset_index(drop=True)
            sub_datas.append(_data)
        return sub_datas

    @classmethod
    def split_by_percentage(cls, data, splits):
        assert sum(splits) == 1 and all([s <= 1.0 for s in splits])
        N = len(data)
        splits = [int(N * _s) for _s in splits]
        splits[-1] = N - sum(splits[:-1])
        return cls.split_by_number(data, splits)

    @classmethod
    def split_df(cls, data, splits):
        if any([s < 1 for s in splits]):
            return cls.split_by_percentage(data, splits)
        else:
            return cls.split_by_number(data, splits)


# %% tags=["active-ipynb", "style-student"]
# data = pd.DataFrame(list(range(10)))
#
# splits1 = [0.5, 0.2, 0.3]
# splits2 = [5, 2, 3]
#
# sub_datas1 = DF_spliter.split_df(data, splits1)
# sub_datas2 = DF_spliter.split_df(data, splits1)
# for d1, d2 in zip(sub_datas1, sub_datas2):
#     print(d1 == d2)

# %%
# Function to format the right-hand side of slashes with consistent length for each column
def format_numeric_of_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the right-hand side of slashes with consistent length for each column in a pandas DataFrame.

    This function takes a pandas DataFrame as input and returns a new DataFrame where the right-hand side
    of slashes in each column has been formatted to have consistent lengths. The maximum length is determined
    by finding the longest value on the right side of '/' for each column, and then padding shorter values
    with phantom zeros.

    Args:
        df (pd.DataFrame): Input pandas DataFrame

    Returns:
        pd.DataFrame: Formatted pandas DataFrame

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'L1': {'0': '99.96/10.90', '1': '95.00/6.94'}, 'D1': {'0': '99.96/100.90', '1': '99.37/3.68'}, 'D2': {'0': '99.88/1.43', '1': '99.32/31.85'}})
        >>> formatted_df = format_phantom_by_column(df)
        >>> print(formatted_df)
                                L1                                  D1                        D2
        0              99.96/10.90                        99.96/100.90   99.88/$\phantom{0}$1.43 
        1  95.00/$\phantom{0}$6.94  99.37/$\phantom{0}\phantom{0}$3.68               99.32/31.85

    Notes:
        - This function assumes that the right-hand side of slashes in each value is numeric.
        - Phantom zeros are used to pad shorter values, which may not be suitable for all use cases.
    """

    def format_column(column: pd.Series) -> pd.Series:
        # Step 1: Find the max length of numbers on the right side of '/'
        max_len = max([len(x.split("/")[1]) if "/" in str(x) else 0 for x in column])
        # Step 2: Format values in the column
        def format_value(val: str) -> str:
            val = str(val)
            if "/" in val:
                left, right = val.split("/")
                # Pad with phantom zeros to the max length for the column
                if (r:= len(right)) < max_len:
                    right_padded = "$" + r"\phantom{0}" * (max_len-r) + "$"  + right
                else:
                    right_padded = right
                return f"{left}/{right_padded}"
            return val

        # Apply formatting to each value in the column
        return column.apply(format_value)

    # Apply the formatting function to each column in the DataFrame
    return df.apply(format_column)


# %%
def check_same_labels_for_duplicated_column(df, column1, column2='label'):
    """
    Check for duplicate `column1` entries in DataFrame and verify if their `column2` values are consistent.

    Args:
        df (DataFrame): A pandas DataFrame containing at least columns 'filename' and 'label'.
        column1 (str): The name of the column to check for duplicates.
        column2 (str, optional): The name of the column to check for consistency. Defaults to 'label'.

    Returns:
        None

    Prints messages indicating whether any inconsistent duplicates were found.

    Raises:
        TypeError: If `df` is not a pandas DataFrame or if `column1` and/or `column2` are not strings.
        KeyError: If `df` does not contain columns `column1` and/or `column2`.

    Example:
        >>import pandas as pd
        >>data = {'filename': ['file1', 'file2', 'file3', 'file4'],
                >>       'label': ['A', 'B', 'C', 'D']}
        >>df = pd.DataFrame(data)
        >>check_same_labels_for_duplicated_column(df, 'filename')
        All duplicate filenames have consistent labels.

        >>import pandas as pd
        >>data = {'filename': ['file1', 'file2', 'file1', 'file4', 'file5'],
                >>       'label': ['A', 'B', 'C', 'D', 'E']}
        >>df = pd.DataFrame(data)
        >>check_same_labels_for_duplicated_column(df, 'filename')
        All duplicate filenames have consistent labels.

    """
    
    # Validate input data types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input `df` must be a pandas DataFrame.")
    if not isinstance(column1, str) or (column2 and not isinstance(column2, str)):
        raise TypeError("Both `column1` and `column2` must be strings.")

    # Validate the existence of columns in the dataframe
    required_columns = [column1, column2]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column `{col}` not found in DataFrame.")

    # Find duplicate filenames and check if labels are consistent
    duplicates = df[df.duplicated(subset='filename', keep=False)]
    consistent_labels = duplicates.groupby('filename')['label'].nunique() == 1


    # Find duplicate data on column1
    duplicated_data = df[df.duplicated(subset=column1, keep=False)]
    
    # Group by 'column1' and check if there are multiple unique 'column2' values for each row
    has_inconsistent_labels = duplicated_data.groupby(column1)[column2].nunique() > 1
    
    # Get inconsistent duplicate files
    inconsistent_files = has_inconsistent_labels[has_inconsistent_labels].index
    
    if not inconsistent_files.empty:
        print("Inconsistent duplicates found for the following filenames:")
        print(inconsistent_files.tolist())
    else:
        print(f"All duplicate \'{column1}\' have consistent \'{column2}\' values.")
