import asyncio
import sys
import pandas as pd
import re
import numpy as np
sys.path.append('.')

from src.utils.data_ingestion import DataIngestionManager, SUPPORTED_CSV_TYPES

async def debug_ingestion():
    # Read the CSV
    df = pd.read_csv('data_sources/fedex_pricing.csv')
    print(f"Original CSV rows: {len(df)}")
    
    # Simulate the preprocessing
    csv_type = "fedex_pricing"
    config = SUPPORTED_CSV_TYPES[csv_type]
    column_mapping = config["column_mapping"]
    
    # Preprocess weight
    def parse_weight(val):
        try:
            val = str(val).strip().lower().replace('"', '').replace(",", "")
            if "oz" in val:
                num = float(re.findall(r"[\d.]+", val)[0])
                return round(num / 16, 4)  # 16 oz = 1 lb
            elif "lb" in val or "lbs" in val:
                num = float(re.findall(r"[\d.]+", val)[0])
                return num
            else:
                return float(val)
        except Exception as e:
            print(f"Could not parse weight '{val}': {e}")
            return np.nan
    
    # Preprocess price
    def parse_price(val):
        try:
            val = str(val).replace("$", "").replace(",", "").strip()
            return float(val)
        except Exception as e:
            print(f"Could not parse price '{val}': {e}")
            return np.nan
    
    # Apply preprocessing
    weight_col = [k for k, v in column_mapping.items() if v == "weight"][0]
    df[weight_col] = df[weight_col].apply(parse_weight)
    
    price_col = [k for k, v in column_mapping.items() if v == "price"][0]
    df[price_col] = df[price_col].apply(parse_price)
    
    print(f"After preprocessing: {len(df)} rows")
    
    # Check for NaN values
    required_cols = [k for k, v in column_mapping.items()]
    nan_counts = df[required_cols].isna().sum()
    print(f"NaN counts per column: {nan_counts.to_dict()}")
    
    # Find rows with NaN in price column
    nan_price_rows = df[df[price_col].isna()]
    print(f"\nRows with NaN in price column ({len(nan_price_rows)} rows):")
    for idx, row in nan_price_rows.head(20).iterrows():  # Show first 20
        print(f"Row {idx + 2}: {dict(row)}")  # +2 for 1-based indexing and header
    
    if len(nan_price_rows) > 20:
        print(f"... and {len(nan_price_rows) - 20} more rows with NaN prices")
    
    # Show original values for NaN price rows
    print(f"\nOriginal price values for NaN rows:")
    original_df = pd.read_csv('data_sources/fedex_pricing.csv')
    for idx in nan_price_rows.head(10).index:
        original_row = original_df.iloc[idx]
        print(f"Row {idx + 2}: Original price = '{original_row['price']}'")
    
    # Drop rows with NaN
    df_clean = df.dropna(subset=required_cols)
    print(f"\nAfter dropping NaN rows: {len(df_clean)} rows")
    
    # Show some sample data
    print("\nSample preprocessed data:")
    print(df_clean.head())

if __name__ == "__main__":
    asyncio.run(debug_ingestion()) 