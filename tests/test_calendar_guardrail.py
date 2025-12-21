import pandas as pd
import pytest
from ML.utils import validate_trading_calendar

def test_valid_calendar():
    # Create valid trading calendar (Mon-Fri)
    dates = pd.date_range(start="2023-01-01", periods=10, freq="B") # Business days
    df = pd.DataFrame({"col": range(10)}, index=dates)
    
    try:
        validate_trading_calendar(df)
        print("Valid calendar passed.")
    except Exception as e:
        pytest.fail(f"Valid calendar failed: {e}")

def test_weekend_error():
    # Include a Saturday
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D") # Daily includes weekends
    df = pd.DataFrame({"col": range(10)}, index=dates)
    
    with pytest.raises(ValueError, match="weekend"):
        validate_trading_calendar(df)
    print("Weekend error caught correctly.")

def test_duplicate_error():
    dates = pd.to_datetime(["2023-01-03", "2023-01-03", "2023-01-04"])
    df = pd.DataFrame({"col": range(3)}, index=dates)
    
    with pytest.raises(ValueError, match="duplicates"):
        validate_trading_calendar(df)
    print("Duplicate error caught correctly.")

def test_monotonic_error():
    dates = pd.to_datetime(["2023-01-04", "2023-01-03"])
    df = pd.DataFrame({"col": range(2)}, index=dates)
    
    with pytest.raises(ValueError, match="monotonic"):
        validate_trading_calendar(df)
    print("Monotonic error caught correctly.")

if __name__ == "__main__":
    test_valid_calendar()
    test_weekend_error()
    test_duplicate_error()
    test_monotonic_error()
    print("All guardrail tests passed!")
