"""
Component for Sentiment Analysis
"""
import pandas as pd
import flair
import torch
from tqdm import tqdm

# Configure tqdm to work with pandas
tqdm.pandas()

def get_sentiment_flair(text, model):
    """
    Analyzes the sentiment of a given text using a pre-loaded Flair model.
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0.0, 0.0
    
    sentence = flair.data.Sentence(str(text)[:512]) # Truncate for performance
    model.predict(sentence)
    
    if sentence.labels:
        label = sentence.labels[0]
        polarity = label.score if label.value == 'POSITIVE' else -label.score
        return polarity, 0.0 # Return polarity and 0.0 for subjectivity
    else:
        return 0.0, 0.0

def analyze_sentiment(df):
    """
    Performs sentiment analysis on the 'description' column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with added 'sentiment_polarity' and 'sentiment_subjectivity' columns.
    """
    print("\n" + "-" * 80)
    print("RUNNING SENTIMENT ANALYSIS")
    print("-" * 80)

    # Set Flair to use GPU if available
    if torch.cuda.is_available():
        flair.device = torch.device('cuda')
        print("  Flair is using GPU.")
    else:
        flair.device = torch.device('cpu')
        print("  Flair is using CPU (CUDA not available).")

    print("  Loading Flair sentiment model (this may take a moment)...")
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    print("  Model loaded. Starting analysis...")

    # Use progress_apply for a clean progress bar
    results = df['description'].progress_apply(
        lambda x: pd.Series(get_sentiment_flair(x, sentiment_model))
    )
    df[['sentiment_polarity', 'sentiment_subjectivity']] = results
    
    print("  \u2713 Sentiment analysis complete.")
    return df
