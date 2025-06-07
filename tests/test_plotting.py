from plotting import standardize_time_period
import pandas as pd

def test_standardize_time_period():
    assert standardize_time_period(('2023-04-25', '2025-02-26')) == (pd.Timestamp(2023,4,25).normalize(), pd.Timestamp(2025,2,26).normalize())

def test_standardize_time_period_season():
    assert standardize_time_period('spring-2024') == (pd.Timestamp(2024,3,20).normalize(), pd.Timestamp(2024,6,20).normalize())
    assert standardize_time_period('summer-2024') == (pd.Timestamp(2024,6,21).normalize(), pd.Timestamp(2024,9,22).normalize())
    assert standardize_time_period('autumn-1980') == (pd.Timestamp(1980,9,23).normalize(), pd.Timestamp(1980,12,20).normalize())
    assert standardize_time_period('winter-2024') == (pd.Timestamp(2024,12,21).normalize(), pd.Timestamp(2025,3,19).normalize())