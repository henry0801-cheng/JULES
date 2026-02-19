import pandas as pd

try:
    xl = pd.ExcelFile('0210-1 RSI/cleaned_stock_data2.xlsx')
    print(f"Sheets: {xl.sheet_names}")
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, nrows=5)
        print(f"\nSheet {sheet} columns: {list(df.columns)}")
        print(f"Sheet {sheet} head:\n{df.head()}")
except Exception as e:
    print(f"Error reading 0210-1 RSI/cleaned_stock_data2.xlsx: {e}")

try:
    xl = pd.ExcelFile('Clean data 0209/cleaned_stock_data1.xlsx')
    print(f"Sheets: {xl.sheet_names}")
except Exception as e:
    print(f"Error reading Clean data 0209/cleaned_stock_data1.xlsx: {e}")
