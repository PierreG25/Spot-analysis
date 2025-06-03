from src.plotting import process_time_period

def test_process_time_period():
    assert 1==1
    assert process_time_period('spring-2024') == ('2024-03-20', '2024-06-20')
    assert process_time_period('spring-2024') == ('2024-06-21', '2024-09-22')
    assert process_time_period('spring-2024') == ('2024-09-23', '2024-12-20')
    assert process_time_period('spring-2024') == ('2024-03-20', '2025-03-19')