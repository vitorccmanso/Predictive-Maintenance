import pandas as pd

def calculate_whiskers(column):
    """
    Calculate upper and lower whiskers for identifying potential outliers in a dataset

    Parameters:
    - column: The column for which whiskers are to be calculated

    Returns:
    - upper_whisker: The upper whisker value
    - lower_whisker: The lower whisker value
    """
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    upper_whisker = q3 + 1.5 * (q3-q1)
    lower_whisker = q1 - 1.5 * (q3-q1)
    
    return upper_whisker, lower_whisker