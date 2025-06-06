from plotting import standardize_time_period
from datetime import date

def test_standardize_time_period():
    assert standardize_time_period(('2023-04-25', '2025-02-26')) == (date(2023,4,25), date(2025,2,26))

def test_standardize_time_period_season():
    assert standardize_time_period('spring-2024') == (date(2024,3,20), date(2024,6,20))
    assert standardize_time_period('summer-2024') == (date(2024,6,21), date(2024,9,22))
    assert standardize_time_period('autumn-1980') == (date(1980,9,23), date(1980,12,20))
    assert standardize_time_period('winter-2024') == (date(2024,12,21), date(2025,3,19))