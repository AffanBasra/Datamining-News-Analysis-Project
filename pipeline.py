"""
Main Pipeline - Orchestrates all stages
Run: python pipeline.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.ingest import ingest_data
from src.preprocess import preprocess_data
from src.visualize import visualize_data
from src.config import COMBINED_CSV


def run_pipeline(save_output=True):
    """
    Execute the complete news analysis pipeline

    Pipeline stages:
    1. Ingest - Load Dawn and Tribune CSV data
    2. Preprocess - Clean and normalize categories
    3. Visualize - Generate analysis charts

    Args:
        save_output: Whether to save cleaned data to CSV
    """
    print("=" * 80)
    print("NEWS ANALYSIS PIPELINE")
    print("=" * 80)

    # Stage 1: Ingestion
    df = ingest_data()

    # Stage 2: Preprocessing
    df_clean = preprocess_data(df)

    # Save cleaned data
    if save_output:
        print("\n" + "=" * 80)
        print("SAVING PROCESSED DATA")
        print("=" * 80)
        print(f"\nSaving cleaned data to {COMBINED_CSV}...")
        df_clean.to_csv(COMBINED_CSV, index=False, encoding='utf-8-sig')
        print(f"✓ Saved {len(df_clean):,} records")

    # Stage 3: Visualization
    visualize_data(df_clean)

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nTotal articles processed: {len(df_clean):,}")
    print(f"Output file: {COMBINED_CSV}")
    print(f"Visualizations: outputs/figures/")
    print("\nGenerated visualizations:")
    print("  - story_length_boxplot.png")
    print("  - source_category_heatmap.png")
    print("  - temporal_trend_lineplot.png")

    return df_clean


if __name__ == "__main__":
    df_final = run_pipeline(save_output=True)
