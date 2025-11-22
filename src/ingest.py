"""
Stage 1: Ingestion - Load raw CSV data from Dawn and Tribune
"""
import pandas as pd
import sys
from src.config import DAWN_CSV, TRIBUNE_CSV

sys.stdout.reconfigure(encoding='utf-8')


def ingest_data():
    """
    Load Dawn and Tribune datasets and combine them

    Returns:
        DataFrame: Combined raw data
    """
    print("=" * 80)
    print("STAGE 1: INGESTION")
    print("=" * 80)

    # Load Dawn dataset
    print("\n[1/3] Loading Dawn dataset...")
    dawn_df = pd.read_csv(DAWN_CSV, encoding="latin-1", low_memory=False)
    # Keep only meaningful columns
    dawn_df = dawn_df[['headline', 'date', 'link', 'source', 'categories', 'description']]
    print(f"  Dawn: {dawn_df.shape[0]:,} articles")

    # Load Tribune dataset
    print("\n[2/3] Loading Tribune dataset...")
    tribune_df = pd.read_csv(TRIBUNE_CSV, encoding='latin-1', low_memory=False)
    print(f"  Tribune: {tribune_df.shape[0]:,} articles")

    # Concatenate
    print("\n[3/3] Combining datasets...")
    combined_df = pd.concat([dawn_df, tribune_df], axis=0, ignore_index=True)
    print(f"  Combined: {combined_df.shape[0]:,} total articles")

    print("\n✓ Ingestion complete")
    return combined_df


if __name__ == "__main__":
    df = ingest_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
