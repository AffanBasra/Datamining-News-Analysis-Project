"""
Stage 2: Base Preprocessing - Clean categories and normalize data
"""
import pandas as pd
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')


def clean_category(cat):
    """Clean and normalize a single category value"""
    if pd.isna(cat):
        return 'Unknown'

    cat = str(cat).strip()

    if len(cat) > 50:
        return 'Unknown'

    cat_lower = cat.lower().strip()

    # Apply normalization rules
    if cat_lower in ['sport', 'sports', 'sport ', 'sports ']:
        return 'Sports'
    elif cat_lower in ['tech', 'technology', 'tech ', 'technology ']:
        return 'Technology'
    elif cat_lower in ['pakistan', 'pakistan ', 'pak']:
        return 'Pakistan'
    elif cat_lower in ['world', 'world ', 'international', 'international ']:
        return 'World'
    elif cat_lower in ['business', 'business ', 'economy', 'finance', 'economy ', 'finance ']:
        return 'Business'
    elif cat_lower in ['entertainment', 'entertainment ', 'showbiz', 'lifestyle', 'culture',
                       'life & style', 'film', 'tv', 'music', 'gossip']:
        return 'Entertainment'
    elif cat_lower in ['opinion', 'opinion ', 'editorial', 'blogs', 'letters',
                       'editorial ', 'blogs ', 'letters ']:
        return 'Opinion'
    elif cat_lower in ['politics', 'political', 'politics ', 'political ']:
        return 'Politics'
    # ... (add more rules as needed) ...
    else:
        return ' '.join(cat.split()).title()


def clean_text(text):
    """
    Performs basic text cleaning.
    - Removes HTML tags
    - Converts to lowercase
    - Removes special characters (keeps letters, numbers, and spaces)
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def preprocess_data(df, sample_size=None):
    """
    Performs base preprocessing on the combined dataset.
    - Cleans categories
    - Cleans description text for topic modeling

    Args:
        df (pd.DataFrame): Raw combined DataFrame.
        sample_size (int, optional): If provided, runs on a random sample of this size.

    Returns:
        pd.DataFrame: DataFrame with 'category_clean' and 'description_clean' columns.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: BASE PREPROCESSING")
    print("=" * 80)

    df = df.copy()

    if sample_size:
        print(f"\nUsing a random sample of {sample_size} articles for preprocessing.")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    print("\n[1/2] Cleaning and normalizing categories...")
    df['categories'] = df['categories'].str.strip()
    df['category_clean'] = df['categories'].apply(clean_category)
    print(f"  ✓ Category cleaning complete.")

    print("\n[2/2] Performing basic text cleaning for topic modeling...")
    df['description_clean'] = df['description'].apply(clean_text)
    print(f"  ✓ Text cleaning complete.")
    
    print("\n✓ Base preprocessing complete")
    return df

