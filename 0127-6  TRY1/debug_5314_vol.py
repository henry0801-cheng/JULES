import pandas as pd
import numpy as np

try:
    print("Loading Q...")
    df_q = pd.read_excel('cleaned_stock_data1.xlsx', sheet_name='Q')
    df_q['Date'] = pd.to_datetime(df_q['Date'])
    df_q.set_index('Date', inplace=True)

    stock = '5314_世紀*'

    # Check data start
    valid_data_q = df_q[stock].dropna()
    print(f"\n{stock} Volume Start: {valid_data_q.index[0]}")
    print(f"{stock} Volume End: {valid_data_q.index[-1]}")
    print(f"Total Volume Points: {len(valid_data_q)}")

    # Calculate MA
    vol_ma = df_q[stock].rolling(window=20).mean()

    target_date = pd.Timestamp('2023-12-18')
    target_idx = df_q.index.get_loc(target_date)
    print(f"\nTarget Date {target_date} Index: {target_idx}")

    if target_date in vol_ma.index:
        ma_val = vol_ma.loc[target_date]
        print(f"Volume MA on {target_date}: {ma_val}")
        print(f"Is NaN? {pd.isna(ma_val)}")

        # Check previous days to see when it becomes valid
        print("\nRecent MA values:")
        print(vol_ma.loc[:target_date].tail(5))
    else:
        print(f"Target date not in MA index.")

except Exception as e:
    print(f"Error: {e}")
