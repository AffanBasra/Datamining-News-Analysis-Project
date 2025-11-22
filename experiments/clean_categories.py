import pandas as pd
import numpy as np
import sys
import re

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Load the combined dataset
print("Loading combined dataset...")
combined_df = pd.read_csv(
    "C:\\Users\\DELL\\Desktop\\Data Mining\\DataMiningProject\\combined_news_data.csv",
    encoding='utf-8-sig'
)

print(f"Dataset shape: {combined_df.shape}")

# Analyze problematic categories
print("\n" + "=" * 80)
print("IDENTIFYING PROBLEMATIC CATEGORIES")
print("=" * 80)

# Find categories that are too long (likely parsing errors)
long_categories = combined_df[combined_df['categories'].str.len() > 50]
print(f"\nCategories with length > 50 characters: {len(long_categories)}")
if len(long_categories) > 0:
    print("\nSample of problematic categories:")
    for idx, row in long_categories.head(10).iterrows():
        print(f"  Row {idx}: '{row['categories'][:100]}...'")

# Enhanced category normalization
print("\n" + "=" * 80)
print("ENHANCED CATEGORY NORMALIZATION")
print("=" * 80)

def clean_category(cat):
    """
    Clean and normalize category names
    """
    if pd.isna(cat):
        return 'Unknown'

    # Convert to string and strip whitespace
    cat = str(cat).strip()

    # If category is too long (likely a parsing error), mark as Unknown
    if len(cat) > 50:
        return 'Unknown'

    # Convert to lowercase for comparison
    cat_lower = cat.lower().strip()

    # Sports variations
    if cat_lower in ['sport', 'sports', 'sport ', 'sports ']:
        return 'Sports'

    # Technology variations
    elif cat_lower in ['tech', 'technology', 'tech ', 'technology ']:
        return 'Technology'

    # Pakistan variations
    elif cat_lower in ['pakistan', 'pakistan ', 'pak']:
        return 'Pakistan'

    # World/International variations
    elif cat_lower in ['world', 'world ', 'international', 'international ']:
        return 'World'

    # Business variations
    elif cat_lower in ['business', 'business ', 'economy', 'finance', 'economy ', 'finance ']:
        return 'Business'

    # Entertainment variations
    elif cat_lower in ['entertainment', 'entertainment ', 'showbiz', 'lifestyle', 'culture', 'life & style', 'film', 'tv', 'music', 'gossip']:
        return 'Entertainment'

    # Opinion variations
    elif cat_lower in ['opinion', 'opinion ', 'editorial', 'blogs', 'letters', 'editorial ', 'blogs ', 'letters ']:
        return 'Opinion'

    # Politics
    elif cat_lower in ['politics', 'political', 'politics ', 'political ']:
        return 'Politics'

    # Health
    elif cat_lower in ['health', 'health ', 'healthcare', 'medicine']:
        return 'Health'

    # Education
    elif cat_lower in ['education', 'education ', 'learning']:
        return 'Education'

    # Science
    elif cat_lower in ['science', 'science ', 'research']:
        return 'Science'

    # Regional categories - keep as is but clean
    elif cat_lower in ['punjab', 'sindh', 'balochistan', 'k-p', 'kp', 'khyber pakhtunkhwa', 'islamabad', 'kashmir', 'gilgit-baltistan']:
        return cat.strip().title()

    # Sports subcategories
    elif cat_lower in ['cricket', 'football', 'hockey', 'tennis', 'athletics']:
        return 'Sports'

    # If it contains comma, it's a multi-category - take the first one
    elif ',' in cat:
        first_cat = cat.split(',')[0].strip()
        return clean_category(first_cat)  # Recursively clean the first category

    # If it looks like a sentence (has multiple spaces), mark as Unknown
    elif cat.count(' ') > 3:
        return 'Unknown'

    # Otherwise, clean and title case
    else:
        # Remove extra spaces
        cleaned = ' '.join(cat.split())
        return cleaned.title()

# Apply enhanced cleaning
combined_df['category_clean'] = combined_df['categories'].apply(clean_category)

# Show results
print("\n" + "=" * 80)
print("CATEGORY DISTRIBUTION AFTER ENHANCED CLEANING")
print("=" * 80)
print(f"\nTotal unique cleaned categories: {combined_df['category_clean'].nunique()}")
print(f"\nTop 30 cleaned categories:")
print(combined_df['category_clean'].value_counts().head(30))

# Compare before and after
print("\n" + "=" * 80)
print("BEFORE vs AFTER COMPARISON")
print("=" * 80)
print(f"Unique categories before: {combined_df['categories'].nunique()}")
print(f"Unique categories after: {combined_df['category_clean'].nunique()}")
print(f"Reduction: {combined_df['categories'].nunique() - combined_df['category_clean'].nunique()} categories consolidated")

# Show distribution of major categories
print("\n" + "=" * 80)
print("MAJOR CATEGORY DISTRIBUTION")
print("=" * 80)
major_categories = ['Pakistan', 'World', 'Business', 'Sports', 'Technology',
                   'Entertainment', 'Opinion', 'Politics', 'Health', 'Education',
                   'Science', 'Unknown']

for cat in major_categories:
    count = (combined_df['category_clean'] == cat).sum()
    percentage = (count / len(combined_df)) * 100
    print(f"{cat:20} {count:6,} ({percentage:5.2f}%)")

# Data quality summary
print("\n" + "=" * 80)
print("FINAL DATA QUALITY SUMMARY")
print("=" * 80)
print(f"Total records: {len(combined_df):,}")
print(f"Missing descriptions: {combined_df['description'].isnull().sum()}")
print(f"Duplicate headlines: {combined_df['headline'].duplicated().sum()}")
print(f"Duplicate links: {combined_df['link'].duplicated().sum()}")
print(f"Unknown categories: {(combined_df['category_clean'] == 'Unknown').sum()}")

# Save the cleaned dataset
output_path = "C:\\Users\\DELL\\Desktop\\Data Mining\\DataMiningProject\\combined_news_data_cleaned.csv"
combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✓ Cleaned dataset saved to: {output_path}")

# Display sample
print("\n" + "=" * 80)
print("SAMPLE OF FINAL CLEANED DATASET")
print("=" * 80)
print(combined_df[['headline', 'date', 'source', 'categories', 'category_clean']].head(15).to_string())

print("\n" + "=" * 80)
print("CLEANING COMPLETE!")
print("=" * 80)
