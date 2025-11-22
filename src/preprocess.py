"""
Stage 2: Preprocessing - Clean categories and normalize data
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')


def clean_category(cat):
    """Clean and normalize a single category value"""
    if pd.isna(cat):
        return 'Unknown'

    cat = str(cat).strip()

    # If too long (parsing error), mark as Unknown
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
    elif cat_lower in ['health', 'health ', 'healthcare', 'medicine']:
        return 'Health'
    elif cat_lower in ['education', 'education ', 'learning']:
        return 'Education'
    elif cat_lower in ['science', 'science ', 'research']:
        return 'Science'
    # Regional categories
    elif cat_lower in ['punjab', 'sindh', 'balochistan', 'k-p', 'kp',
                       'khyber pakhtunkhwa', 'islamabad', 'kashmir', 'gilgit-baltistan']:
        return cat.strip().title()
    # Sports subcategories
    elif cat_lower in ['cricket', 'football', 'hockey', 'tennis', 'athletics']:
        return 'Sports'
    # Multi-category - take first
    elif ',' in cat:
        first_cat = cat.split(',')[0].strip()
        return clean_category(first_cat)
    # Sentence-like text
    elif cat.count(' ') > 3:
        return 'Unknown'
    else:
        return ' '.join(cat.split()).title()


def preprocess_data(df):
    """
    Preprocess the combined dataset

    Args:
        df: Raw combined DataFrame

    Returns:
        DataFrame: Cleaned DataFrame with category_clean column
    """
    print("\n" + "=" * 80)
    print("STAGE 2: PREPROCESSING")
    print("=" * 80)

    df = df.copy()

    # Strip whitespace from categories
    print("\n[1/2] Cleaning categories...")
    df['categories'] = df['categories'].str.strip()

    original_categories = df['categories'].nunique()
    print(f"  Original unique categories: {original_categories}")

    # Apply category normalization
    print("\n[2/2] Applying category normalization...")
    df['category_clean'] = df['categories'].apply(clean_category)

    cleaned_categories = df['category_clean'].nunique()
    print(f"  Cleaned unique categories: {cleaned_categories}")
    print(f"  Reduction: {original_categories - cleaned_categories} categories consolidated")

    print("\n  Top 10 categories after cleaning:")
    for cat, count in df['category_clean'].value_counts().head(10).items():
        pct = (count / len(df)) * 100
        print(f"    {cat:20} {count:6,} ({pct:5.2f}%)")

    print("\n✓ Preprocessing complete")
    return df


if __name__ == "__main__":
    from src.ingest import ingest_data
    df = ingest_data()
    df_clean = preprocess_data(df)
    print(f"\nFinal shape: {df_clean.shape}")
