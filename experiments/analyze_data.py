import pandas as pd
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Load Dawn dataset
dawn_df=pd.read_csv("C:\\Users\\DELL\\Desktop\\Data Mining\\Data Mining Project\\News Datasets\\Copy of Copy of dawn (full-data).csv",encoding="latin-1", low_memory=False)
print("=" * 80)
print("DAWN DATASET OVERVIEW")
print("=" * 80)
print(f"\nShape: {dawn_df.shape} (rows, columns)")
print(f"\nMeaningful Columns (first 6):")
print(dawn_df.columns[:6].tolist())
print(f"\nRemaining columns: {len(dawn_df.columns) - 6} unnamed columns (mostly empty)")

# Check how many unnamed columns have data
unnamed_cols = [col for col in dawn_df.columns if 'Unnamed' in str(col)]
print(f"\nUnnamed columns with non-null values:")
for col in unnamed_cols[:10]:  # Show first 10
    non_null = dawn_df[col].notna().sum()
    if non_null > 0:
        print(f"  {col}: {non_null} non-null values")

print("\n" + "-" * 80)
print("DAWN - Main Columns Analysis")
print("-" * 80)
for col in dawn_df.columns[:6]:
    print(f"\n{col}:")
    print(f"  Non-null: {dawn_df[col].notna().sum()}/{len(dawn_df)}")
    print(f"  Unique values: {dawn_df[col].nunique()}")
    if col in ['categories', 'source']:
        print(f"  Top values: {dawn_df[col].value_counts().head(5).to_dict()}")

# Sample data
print("\n" + "-" * 80)
print("DAWN - Sample Records")
print("-" * 80)
for i in range(min(3, len(dawn_df))):
    print(f"\nRecord {i+1}:")
    for col in dawn_df.columns[:6]:
        val = str(dawn_df[col].iloc[i])[:100]  # Truncate long values
        print(f"  {col}: {val}")

# Load Tribune dataset
tribune_df=pd.read_csv("C:\\Users\\DELL\\Desktop\\Data Mining\\Data Mining Project\\News Datasets\\Copy of Copy of tribune(full-data).csv",encoding='latin-1', low_memory=False)
print("\n\n" + "=" * 80)
print("TRIBUNE DATASET OVERVIEW")
print("=" * 80)
print(f"\nShape: {tribune_df.shape} (rows, columns)")
print(f"\nColumns ({len(tribune_df.columns)}):")
print(tribune_df.columns.tolist())

print("\n" + "-" * 80)
print("TRIBUNE - Columns Analysis")
print("-" * 80)
for col in tribune_df.columns:
    print(f"\n{col}:")
    print(f"  Non-null: {tribune_df[col].notna().sum()}/{len(tribune_df)}")
    print(f"  Unique values: {tribune_df[col].nunique()}")
    if col in ['categories', 'source']:
        print(f"  Top values: {tribune_df[col].value_counts().head(5).to_dict()}")

# Sample data
print("\n" + "-" * 80)
print("TRIBUNE - Sample Records")
print("-" * 80)
for i in range(min(3, len(tribune_df))):
    print(f"\nRecord {i+1}:")
    for col in tribune_df.columns:
        val = str(tribune_df[col].iloc[i])[:100]  # Truncate long values
        print(f"  {col}: {val}")

print("\n\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Dawn records: {len(dawn_df):,}")
print(f"Tribune records: {len(tribune_df):,}")
print(f"Total records: {len(dawn_df) + len(tribune_df):,}")
print(f"\nCommon columns: {set(dawn_df.columns[:6]) & set(tribune_df.columns)}")
print(f"\nDate ranges:")
if pd.api.types.is_numeric_dtype(dawn_df['date']):
    print(f"  Dawn: {dawn_df['date'].min()} to {dawn_df['date'].max()}")
else:
    print(f"  Dawn: {dawn_df['date'].iloc[0]} to {dawn_df['date'].iloc[-1]} (sample)")
if pd.api.types.is_numeric_dtype(tribune_df['date']):
    print(f"  Tribune: {tribune_df['date'].min()} to {tribune_df['date'].max()}")
else:
    print(f"  Tribune: {tribune_df['date'].iloc[0]} to {tribune_df['date'].iloc[-1]} (sample)")
