import pandas as pd
import numpy as np
from typing import List

def custom_groupby(df: pd.DataFrame, groupby_cols: List[str], agg_dict: dict) -> pd.DataFrame:
    """
    Generalized function to group by specified columns and aggregate data.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to group.
    groupby_cols (list): List of column names to group by.
    agg_dict (dict): Dictionary specifying the aggregation functions for each column.
                     Example: {'column_name': 'mean', 'other_column': 'sum'}
    
    Returns:
    pd.DataFrame: The grouped and aggregated DataFrame.
    """
    return df.groupby(groupby_cols).agg(agg_dict).reset_index()

def find_outliers_iqr(df: pd.DataFrame, column: str) -> List:
    """
    Identifies outliers in a numerical column using the IQR method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to check for outliers.

    Returns:
    pd.DataFrame: A DataFrame containing only the outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return [lower_bound, upper_bound]
