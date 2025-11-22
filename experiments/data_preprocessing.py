import pandas as pd
import numpy as np
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Load both datasets
print("Loading datasets...")
dawn_df = pd.read_csv(
    "C:\\Users\\DELL\\Desktop\\Data Mining\\Data Mining Project\\News Datasets\\Copy of Copy of dawn (full-data).csv",
    encoding="latin-1",
    low_memory=False
)

tribune_df = pd.read_csv(
    "C:\\Users\\DELL\\Desktop\\Data Mining\\Data Mining Project\\News Datasets\\Copy of Copy of tribune(full-data).csv",
    encoding='latin-1',
    low_memory=False
)

# Keep only the meaningful columns from Dawn (drop all Unnamed columns)
dawn_df = dawn_df[['headline', 'date', 'link', 'source', 'categories', 'description']]

print(f"Dawn shape after cleaning: {dawn_df.shape}")
print(f"Tribune shape: {tribune_df.shape}")

# Concatenate both datasets
print("\nConcatenating datasets...")
combined_df = pd.concat([dawn_df, tribune_df], axis=0, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

# Check categories before normalization
print("\n" + "=" * 80)
print("CATEGORY ANALYSIS BEFORE NORMALIZATION")
print("=" * 80)
print(f"\nTotal unique categories: {combined_df['categories'].nunique()}")
print(f"\nTop 20 categories:")
print(combined_df['categories'].value_counts().head(20))

# Strip whitespace from categories
combined_df['categories'] = combined_df['categories'].str.strip()

# Create category mapping for normalization
print("\n" + "=" * 80)
print("CATEGORY NORMALIZATION")
print("=" * 80)

# Get all unique categories
unique_categories = combined_df['categories'].unique()
print(f"\nAnalyzing {len(unique_categories)} unique categories...")

# Define normalization rules
category_mapping = {}

for cat in unique_categories:
    if pd.isna(cat):
        category_mapping[cat] = 'Unknown'
    else:
        cat_lower = cat.lower().strip()

        # Sports variations
        if cat_lower in ['sport', 'sports', 'sport ']:
            category_mapping[cat] = 'Sports'

        # Technology variations
        elif cat_lower in ['tech', 'technology', 'tech ', 'technology ']:
            category_mapping[cat] = 'Technology'

        # Pakistan variations
        elif cat_lower in ['pakistan', 'pakistan ', 'pak']:
            category_mapping[cat] = 'Pakistan'

        # World variations
        elif cat_lower in ['world', 'world ', 'international']:
            category_mapping[cat] = 'World'

        # Business variations
        elif cat_lower in ['business', 'business ', 'economy', 'finance']:
            category_mapping[cat] = 'Business'

        # Entertainment variations
        elif cat_lower in ['entertainment', 'entertainment ', 'showbiz', 'lifestyle', 'culture']:
            category_mapping[cat] = 'Entertainment'

        # Opinion variations
        elif cat_lower in ['opinion', 'opinion ', 'editorial', 'blogs', 'letters']:
            category_mapping[cat] = 'Opinion'

        # Politics
        elif cat_lower in ['politics', 'political']:
            category_mapping[cat] = 'Politics'

        # Health
        elif cat_lower in ['health', 'health ', 'healthcare', 'medicine']:
            category_mapping[cat] = 'Health'

        # Education
        elif cat_lower in ['education', 'education ', 'learning']:
            category_mapping[cat] = 'Education'

        # Science
        elif cat_lower in ['science', 'science ', 'research']:
            category_mapping[cat] = 'Science'

        # Default: capitalize first letter and clean
        else:
            category_mapping[cat] = cat.strip().title()

# Apply the mapping
combined_df['category_clean'] = combined_df['categories'].map(category_mapping)

# Show mapping results
print("\nSample of category mappings:")
sample_mappings = dict(list(category_mapping.items())[:30])
for original, cleaned in sample_mappings.items():
    if original != cleaned:
        print(f"  '{original}' → '{cleaned}'")

print("\n" + "=" * 80)
print("CATEGORY ANALYSIS AFTER NORMALIZATION")
print("=" * 80)
print(f"\nTotal unique cleaned categories: {combined_df['category_clean'].nunique()}")
print(f"\nTop 20 cleaned categories:")
print(combined_df['category_clean'].value_counts().head(20))

# Show before/after comparison
print("\n" + "=" * 80)
print("BEFORE vs AFTER COMPARISON")
print("=" * 80)
print(f"Unique categories before: {combined_df['categories'].nunique()}")
print(f"Unique categories after: {combined_df['category_clean'].nunique()}")
print(f"Reduction: {combined_df['categories'].nunique() - combined_df['category_clean'].nunique()} categories consolidated")

# Data quality checks
print("\n" + "=" * 80)
print("DATA QUALITY SUMMARY")
print("=" * 80)
print(f"Total records: {len(combined_df):,}")
print(f"\nMissing values:")
print(combined_df.isnull().sum())
print(f"\nDuplicate headlines: {combined_df['headline'].duplicated().sum()}")
print(f"Duplicate links: {combined_df['link'].duplicated().sum()}")

# Save the combined and cleaned dataset
output_path = "C:\\Users\\DELL\\Desktop\\Data Mining\\DataMiningProject\\combined_news_data.csv"
combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✓ Combined dataset saved to: {output_path}")

# Display sample of final data
print("\n" + "=" * 80)
print("SAMPLE OF FINAL DATASET")
print("=" * 80)
print(combined_df[['headline', 'date', 'source', 'categories', 'category_clean']].head(10))

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)
