import pandas as pd
import os

INPUT_FILE = "Clean data 0209/Stock_data_19_23.xlsx"
OUTPUT_FILE = "Clean data 0209/cleaned_stock_data1.xlsx"

def process_sheet(xl, sheet_name):
    print(f"Processing sheet: {sheet_name}")

    # Read raw data
    # Use header=None to read everything as data first, then parse manually
    df_raw = xl.parse(sheet_name, header=None)

    # Extract Tickers (Row 0, Col 5+)
    # iloc uses 0-based indexing
    # Ensure they are strings and strip whitespace
    tickers = df_raw.iloc[0, 5:].astype(str).str.strip().values

    # Extract Names (Row 1, Col 5+)
    names = df_raw.iloc[1, 5:].astype(str).str.strip().values

    # Combine Ticker and Name
    # Format: Ticker_Name
    cols = [f"{t}_{n}" for t, n in zip(tickers, names)]

    # Extract Date (Col 1, Row 2+)
    dates = df_raw.iloc[2:, 1].astype(str).str.strip().values

    # Extract Data (Col 5+, Row 2+)
    # Keep as object first, convert later
    data_values = df_raw.iloc[2:, 5:].values

    # Create DataFrame
    df = pd.DataFrame(data_values, columns=cols)
    df.insert(0, "Date", dates)

    # Convert Date to datetime
    # Format in file is YYYYMMDD (e.g., 20190102)
    # Errors='coerce' will turn invalid dates to NaT
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')

    # Drop rows with invalid dates (NaT)
    df = df.dropna(subset=['Date'])

    # Sort by Date just in case
    df = df.sort_values('Date')

    # Remove duplicates if any (based on Date)
    df = df.drop_duplicates(subset=['Date'])

    # Set Date as index for reindexing
    df = df.set_index('Date')

    # Create full date range (daily) from min to max date
    # This ensures all calendar days are present
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

    # Reindex to full range
    # This adds rows for missing dates with NaN values
    df = df.reindex(full_idx)

    # Forward Fill missing dates (and values)
    # This implements the logic: "If T+2, T+3 are missing... fill with T+1"
    df = df.ffill()

    # Reset index to make Date a column again
    df.index.name = 'Date'
    df = df.reset_index()

    # Remove weekends (Saturday=5, Sunday=6)
    # The requirement implied filling gaps, but standard market data excludes weekends.
    # Matching cleaned_stock_data1 format which excludes weekends.
    df = df[df['Date'].dt.dayofweek < 5]

    # Ensure numeric types for data columns
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    xl = pd.ExcelFile(INPUT_FILE)

    # Create ExcelWriter
    writer = pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl')

    # Process sheets P, Q, F
    target_sheets = ['P', 'Q', 'F']
    processed_count = 0

    for sheet in target_sheets:
        if sheet in xl.sheet_names:
            df = process_sheet(xl, sheet)
            print(f"  Writing sheet {sheet} with shape {df.shape}")
            df.to_excel(writer, sheet_name=sheet, index=False)
            processed_count += 1
        else:
            print(f"Warning: Sheet {sheet} not found in input.")

    if processed_count > 0:
        writer.close()
        print(f"Successfully created {OUTPUT_FILE}")
    else:
        print("No sheets processed.")

if __name__ == "__main__":
    main()
