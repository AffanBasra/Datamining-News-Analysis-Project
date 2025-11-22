"""
Main entry point - Runs the complete pipeline
For quick exploration, use: python main.py
For full pipeline control, use: python pipeline.py
"""
import pandas as pd
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Check if cleaned data exists
CLEANED_DATA = "combined_news_data_cleaned.csv"

if not Path(CLEANED_DATA).exists():
    print("Cleaned data not found. Running pipeline...")
    from pipeline import run_pipeline
    df = run_pipeline(save_output=True)
else:
    print("Loading existing cleaned dataset...")
    df = pd.read_csv(CLEANED_DATA, encoding='utf-8-sig')

    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"\nTotal articles: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    print("\n" + "=" * 80)
    print("CATEGORY DISTRIBUTION")
    print("=" * 80)
    print("\nTop 10 categories:")
    for cat, count in df['category_clean'].value_counts().head(10).items():
        percentage = (count / len(df)) * 100
        print(f"  {cat:20} {count:6,} ({percentage:5.2f}%)")

    print("\n" + "=" * 80)
    print("SOURCE DISTRIBUTION")
    print("=" * 80)
    for source, count in df['source'].value_counts().head(3).items():
        print(f"  {source:20} {count:6,}")

    print("\n" + "=" * 80)
    print("SAMPLE RECORDS")
    print("=" * 80)
    print(df[['headline', 'date', 'source', 'category_clean']].head(10).to_string())

    print("\n" + "=" * 80)
    print("To regenerate everything, run: python pipeline.py")
    print("=" * 80)

