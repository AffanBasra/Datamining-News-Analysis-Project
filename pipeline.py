"""
Main Pipeline - Orchestrates all modular stages of the News Analysis project.
"""
import sys
import os
import pandas as pd
from pathlib import Path

# --- Local Module Imports ---
from src.ingest import ingest_data
from src.preprocess import preprocess_data
from src.sentiment_analysis import analyze_sentiment
from src.topic_modeling import model_topics
from src.ner import perform_ner
from src.visualize import visualize_data
from src.config import COMBINED_CSV, FIGURES_DIR

# --- Main Orchestration Function ---

def run_pipeline(
    sample_size=None,
    force_reprocess_base=False,
    force_reprocess_sentiment=False,
    run_topic_modeling=False,
    run_ner=False,
    save_output=True
):
    """
    Execute the complete, modular news analysis pipeline.

    Args:
        sample_size (int, optional): Runs the pipeline on a random sample of this size.
        force_reprocess_base (bool): Forces re-running of data ingestion and base preprocessing.
        force_reprocess_sentiment (bool): Forces re-running of sentiment analysis, even if columns exist.
        run_topic_modeling (bool): If True, runs the topic modeling step.
        run_ner (bool): If True, runs the Named Entity Recognition step.
        save_output (bool): If True, saves the final processed DataFrame to CSV.
    """
    print("=" * 80)
    print("MODULAR NEWS ANALYSIS PIPELINE")
    # --- Print run configuration ---
    if sample_size: print(f"|--- RUNNING ON A SAMPLE OF {sample_size} ARTICLES")
    if force_reprocess_base: print("|--- FORCING REPROCESSING: BASE")
    if force_reprocess_sentiment: print("|--- FORCING REPROCESSING: SENTIMENT")
    if run_topic_modeling: print("|--- OPTION ENABLED: TOPIC MODELING")
    if run_ner: print("|--- OPTION ENABLED: NAMED ENTITY RECOGNITION")
    print("=" * 80)

    # --- 1. Load or Perform Base Preprocessing ---
    output_csv = COMBINED_CSV.replace('.csv', f'_sample_{sample_size}.csv') if sample_size else COMBINED_CSV
    
    if not force_reprocess_base and os.path.exists(output_csv):
        print(f"\nFound existing file: '{output_csv}'. Loading from cache.")
        df = pd.read_csv(output_csv)
    else:
        print("\nRunning base data ingestion and preprocessing...")
        df_raw = ingest_data()
        df = preprocess_data(df_raw, sample_size=sample_size)

    # --- 2. Conditional Sentiment Analysis ---
    if 'sentiment_polarity' not in df.columns or force_reprocess_sentiment:
        df = analyze_sentiment(df)
    else:
        print("\nSentiment analysis columns already exist. Skipping.")
        print("  To re-run, use the 'force_reprocess_sentiment=True' argument.")

    # --- 3. Conditional Topic Modeling ---
    if run_topic_modeling:
        topic_model_output_dir = Path("outputs/topic_modeling")
        _, df = model_topics(df, topic_model_output_dir) # We only need the df with topic assignments here
    else:
        print("\nSkipping topic modeling.")
        print("  To run this step, use the 'run_topic_modeling=True' argument.")

    # --- 4. Conditional Named Entity Recognition (NER) ---
    if run_ner:
        df = perform_ner(df)
    else:
        print("\nSkipping Named Entity Recognition (NER).")
        print("  To run this step, use the 'run_ner=True' argument.")

    # --- 5. Save Final Output ---
    if save_output:
        print("\n" + "=" * 80)
        print("SAVING FINAL PROCESSED DATA")
        print("=" * 80)
        print(f"\nSaving final enriched data to {output_csv}...")
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✓ Saved {len(df):,} records with columns: {df.columns.tolist()}")

    # --- 6. Visualization ---
    # Visualizations are generated based on the final state of the DataFrame
    visualize_data(df)

    # --- 7. Summary ---
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nTotal articles in final dataset: {len(df):,}")
    print(f"Final columns: {df.columns.tolist()}")
    if save_output: print(f"Output file: {output_csv}")
    print(f"Visualizations: {FIGURES_DIR}")
    if run_topic_modeling: print(f"Topic Modeling outputs: {topic_model_output_dir}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the modular news analysis pipeline.")
    parser.add_argument('--sample_size', type=int, default=None, help='Run the pipeline on a random sample of this size.')
    parser.add_argument('--force_reprocess_base', action='store_true', help='Force re-running of data ingestion and base preprocessing.')
    parser.add_argument('--force_reprocess_sentiment', action='store_true', help='Force re-running of sentiment analysis.')
    parser.add_argument('--run_topic_modeling', action='store_true', help='Run the topic modeling step.')
    parser.add_argument('--run_ner', action='store_true', help='Run the Named Entity Recognition step.')
    
    args = parser.parse_args()

    run_pipeline(
        sample_size=args.sample_size,
        force_reprocess_base=args.force_reprocess_base,
        force_reprocess_sentiment=args.force_reprocess_sentiment,
        run_topic_modeling=args.run_topic_modeling,
        run_ner=args.run_ner
    )


