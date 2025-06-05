from plotting import process_time_period
from datetime import date

def test_process_time_period():
    assert 1==1
    assert process_time_period('spring-2024') == (date(2024,3,20), date(2024,6,20))
    assert process_time_period('summer-2024') == (date(2024,6,21), date(2024,9,22))
    assert process_time_period('autumn-2024') == (date(2024,9,23), date(2024,12,20))
    assert process_time_period('winter-2024') == (date(2024,12,21), date(2025,3,19))