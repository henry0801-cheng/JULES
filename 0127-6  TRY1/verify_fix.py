import pandas as pd
try:
    df = pd.read_excel('trading_results.xlsx', sheet_name='Transactions')

    # Check for 5314_世紀* on 2023-12-18
    target_stock = '5314_世紀*'
    target_date = pd.Timestamp('2023-12-18')

    suspicious_trade = df[(df['Stock'] == target_stock) & (df['Date'] == target_date)]

    if not suspicious_trade.empty:
        print(f"FAILURE: Trade for {target_stock} on {target_date} STILL EXISTS!")
        print(suspicious_trade)
    else:
        print(f"SUCCESS: Trade for {target_stock} on {target_date} is GONE.")

    print(f"\nTotal transactions: {len(df)}")

except Exception as e:
    print(f"Error reading file: {e}")
