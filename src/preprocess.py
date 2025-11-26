"""
Stage 2: Base Preprocessing - Clean categories and normalize data
"""
import pandas as pd
import sys

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


def preprocess_data(df, sample_size=None):
    """
    Performs base preprocessing on the combined dataset (category cleaning).

    Args:
        df (pd.DataFrame): Raw combined DataFrame.
        sample_size (int, optional): If provided, runs on a random sample of this size.

    Returns:
        pd.DataFrame: DataFrame with the 'category_clean' column.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: BASE PREPROCESSING")
    print("=" * 80)

    df = df.copy()

    if sample_size:
        print(f"\nUsing a random sample of {sample_size} articles for preprocessing.")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    print("\n[1/1] Cleaning and normalizing categories...")
    df['categories'] = df['categories'].str.strip()
    df['category_clean'] = df['categories'].apply(clean_category)
    
    print(f"  ✓ Category cleaning complete.")
    print("\n✓ Base preprocessing complete")
    return df

