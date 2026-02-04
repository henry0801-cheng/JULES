import pandas as pd
import numpy as np

try:
    # Load Price
    print("Loading P...")
    df_p = pd.read_excel('cleaned_stock_data1.xlsx', sheet_name='P')
    df_p['Date'] = pd.to_datetime(df_p['Date'])
    df_p.set_index('Date', inplace=True)

    stock = '5314_世紀*'

    # Check data start
    valid_data = df_p[stock].dropna()
    print(f"\n{stock} Valid Data Start: {valid_data.index[0]}")
    print(f"{stock} Valid Data End: {valid_data.index[-1]}")

    # Simulate signal generation for 2023-12-15 (Signal for 12-18)
    # Find idx for 2023-12-15
    dates = df_p.index
    target_date = pd.Timestamp('2023-12-15')

    if target_date in dates:
        idx = df_p.index.get_loc(target_date)
        print(f"\nSignal Date: {target_date} (Index: {idx})")

        lookback = 10
        past_idx = idx - lookback
        past_date = df_p.index[past_idx]
        print(f"Lookback Date: {past_date} (Index: {past_idx})")

        current_price = df_p.iloc[idx][stock]
        past_price = df_p.iloc[past_idx][stock]

        print(f"Current Price ({target_date}): {current_price}")
        print(f"Past Price ({past_date}): {past_price}")

        # Manual Return Calc
        if pd.isna(past_price) or past_price == 0:
            print("Past price is NaN or 0. Should be invalid.")
            ret = np.nan
        else:
            ret = (current_price - past_price) / past_price
            print(f"Return: {ret}")

        # Check DataFrame ops
        current_prices = df_p.iloc[idx]
        past_prices = df_p.iloc[past_idx]
        past_prices = past_prices.replace(0, np.nan)
        returns = (current_prices - past_prices) / past_prices

        print(f"\nDataFrame Op Return for {stock}: {returns[stock]}")

        valid_returns = returns.dropna()
        if stock in valid_returns:
            print(f"FAILURE: {stock} is in valid_returns!")
        else:
            print(f"SUCCESS: {stock} is NOT in valid_returns.")

    else:
        print(f"Date {target_date} not in index.")

except Exception as e:
    print(f"Error: {e}")
