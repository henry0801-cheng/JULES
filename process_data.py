
import pandas as pd
import os

SOURCE_FILE = "Clean data 0209/Stock_data_19_23.xlsx"
TARGET_FILE = "cleaned_stock_data1.xlsx"

def process_sheet(file_path, sheet_name):
    print(f"Processing sheet {sheet_name}...")

    # Read raw data without header
    try:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"Error reading sheet {sheet_name}: {e}")
        return None

    # Extract Tickers (Row 0, Col 5 onwards)
    tickers = df_raw.iloc[0, 5:].fillna('').astype(str).tolist()

    # Extract Names (Row 1, Col 5 onwards)
    names = df_raw.iloc[1, 5:].fillna('').astype(str).tolist()

    # Combine Ticker and Name
    columns = [f"{t}_{n}" for t, n in zip(tickers, names)]

    # Extract Dates (Col 1, starting from Row 2)
    dates = df_raw.iloc[2:, 1]

    # Extract Data (Col 5 onwards, starting from Row 2)
    data = df_raw.iloc[2:, 5:]

    # Create DataFrame
    # Note: data index will align with dates
    df = pd.DataFrame(data.values, index=dates.values, columns=columns)

    # Convert index to datetime
    # The dates in excel might be integers (20190102) or strings
    df.index = pd.to_datetime(df.index, format='%Y%m%d', errors='coerce')
    df.index.name = 'Date'

    # Drop rows where Date is NaT (invalid date)
    df = df[df.index.notna()]

    # Sort by date
    df = df.sort_index()

    if df.empty:
        print(f"Warning: Sheet {sheet_name} is empty after processing.")
        return df

    start_date = df.index.min()
    end_date = df.index.max()
    print(f"  Date range: {start_date.date()} to {end_date.date()}")

    # Create full daily date range
    full_idx = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex to full range
    # This introduces NaNs for missing days
    df = df.reindex(full_idx)

    # Forward fill to propagate last valid observation forward
    df = df.ffill()

    # Remove weekends
    # Saturday=5, Sunday=6
    is_weekend = df.index.dayofweek >= 5
    df_clean = df[~is_weekend]

    # Reset index to make Date a column
    df_clean = df_clean.reset_index()
    df_clean.rename(columns={'index': 'Date'}, inplace=True)

    return df_clean

def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Source file {SOURCE_FILE} not found.")
        return

    sheets = ['P', 'Q', 'F']
    processed_dfs = {}

    for sheet in sheets:
        df = process_sheet(SOURCE_FILE, sheet)
        if df is not None:
            processed_dfs[sheet] = df
            print(f"  Shape: {df.shape}")
        else:
            print(f"Skipping sheet {sheet} due to error.")
            return

    # Write to Excel
    print(f"Writing to {TARGET_FILE}...")
    try:
        with pd.ExcelWriter(TARGET_FILE, engine='openpyxl') as writer:
            for sheet_name, df in processed_dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print("Successfully created cleaned_stock_data1.xlsx")
    except Exception as e:
        print(f"Error writing to excel: {e}")

if __name__ == "__main__":
    main()
