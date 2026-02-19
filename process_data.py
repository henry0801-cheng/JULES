
import pandas as pd
import os
import sys

SOURCE_FILE = "Clean data 0209/Stock_data_19_23.xlsx"
TARGET_FILE = "cleaned_stock_data1.xlsx"

def process_sheet(file_path, sheet_name):
    print(f"Processing sheet {sheet_name} from {file_path}...")

    try:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"Error reading sheet {sheet_name}: {e}")
        return None

    # Tickers: Row 0, Cols 5+
    # Names: Row 1, Cols 5+
    tickers = df_raw.iloc[0, 5:].fillna('').astype(str).tolist()
    names = df_raw.iloc[1, 5:].fillna('').astype(str).tolist()

    # Check for empty F sheet
    if sheet_name == 'F':
        data_check = df_raw.iloc[2:, 5:]
        if data_check.isna().all().all():
            print(f"Warning: Sheet {sheet_name} contains only NaN values. Output will be empty structure.")

    # Create Ticker_Name headers
    # Remove any surrounding whitespace
    tickers = [t.strip() for t in tickers]
    names = [n.strip() for n in names]
    columns = [f"{t}_{n}" for t, n in zip(tickers, names)]

    # Dates: Row 2+, Col 1
    dates = df_raw.iloc[2:, 1]

    # Data: Row 2+, Cols 5+
    data = df_raw.iloc[2:, 5:]

    # Create DataFrame
    df = pd.DataFrame(data.values, index=dates.values, columns=columns)

    # Convert index to datetime
    # Coerce errors to NaT, drop NaT rows
    df.index = pd.to_datetime(df.index, format='%Y%m%d', errors='coerce')
    df.index.name = 'Date'
    df = df[df.index.notna()]
    df = df.sort_index()

    if df.empty:
        print(f"Sheet {sheet_name} is empty after processing dates.")
        return df

    start_date = df.index.min()
    end_date = df.index.max()
    print(f"  Date range: {start_date.date()} to {end_date.date()}")

    # Convert data to numeric
    # This handles any text artifacts like '--' or 'null' by turning them into NaN
    # Then forward fill will handle NaNs correctly
    print("  Converting data to numeric...")
    df = df.apply(pd.to_numeric, errors='coerce')

    # Create full daily date range to fill gaps
    full_idx = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex (introduces NaNs for missing days)
    # This aligns the index to include ALL days (Mon-Sun)
    df = df.reindex(full_idx)

    # Forward fill (fills missing days with previous day's data)
    # Requirement: "T+1~T+4中间的T+2、T+3缺值，以T+1資料填滿"
    # This fills any NaN (whether from reindex or original data) with previous valid value
    print("  Filling gaps (ffill)...")
    df = df.ffill()

    # Remove weekends (Saturday=5, Sunday=6)
    # This effectively removes the weekend rows but keeps the filled weekday rows
    print("  Removing weekends...")
    is_weekend = df.index.dayofweek >= 5
    df_clean = df[~is_weekend]

    # Reset index to make Date a column
    df_clean = df_clean.reset_index()
    df_clean.rename(columns={'index': 'Date'}, inplace=True)

    return df_clean

def main():
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Source file {SOURCE_FILE} not found.")
        sys.exit(1)

    sheets = ['P', 'Q', 'F']
    processed_dfs = {}

    for sheet in sheets:
        df = process_sheet(SOURCE_FILE, sheet)
        if df is not None:
            processed_dfs[sheet] = df
            print(f"  Final Shape: {df.shape}")
        else:
            print(f"Skipping sheet {sheet} due to error.")
            sys.exit(1)

    print(f"Writing to {TARGET_FILE}...")
    try:
        with pd.ExcelWriter(TARGET_FILE, engine='openpyxl') as writer:
            for sheet_name, df in processed_dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Successfully created {TARGET_FILE}")
    except Exception as e:
        print(f"Error writing to excel: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
