"""
Configuration settings for the news analysis pipeline
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data source paths (external)
DAWN_CSV_PATH = r"C:\Users\DELL\Desktop\Data Mining\Data Mining Project\News Datasets\Copy of Copy of dawn (full-data).csv"
TRIBUNE_CSV_PATH = r"C:\Users\DELL\Desktop\Data Mining\Data Mining Project\News Datasets\Copy of Copy of tribune(full-data).csv"

# Processed data paths
COMBINED_RAW_PATH = PROCESSED_DATA_DIR / "combined_news_raw.csv"
COMBINED_CLEANED_PATH = PROCESSED_DATA_DIR / "combined_news_cleaned.csv"

# Analysis settings
TOP_CATEGORIES = ['Pakistan', 'World', 'Sports', 'Business', 'Entertainment', 'Technology']

# Category normalization mapping
CATEGORY_MAPPINGS = {
    # Sports variations
    'sport': 'Sports',
    'sports': 'Sports',
    'sport ': 'Sports',
    'sports ': 'Sports',
    'cricket': 'Sports',
    'football': 'Sports',
    'hockey': 'Sports',
    'tennis': 'Sports',
    'athletics': 'Sports',

    # Technology variations
    'tech': 'Technology',
    'technology': 'Technology',
    'tech ': 'Technology',
    'technology ': 'Technology',

    # Pakistan variations
    'pakistan': 'Pakistan',
    'pakistan ': 'Pakistan',
    'pak': 'Pakistan',

    # World variations
    'world': 'World',
    'world ': 'World',
    'international': 'World',
    'international ': 'World',

    # Business variations
    'business': 'Business',
    'business ': 'Business',
    'economy': 'Business',
    'economy ': 'Business',
    'finance': 'Business',
    'finance ': 'Business',

    # Entertainment variations
    'entertainment': 'Entertainment',
    'entertainment ': 'Entertainment',
    'showbiz': 'Entertainment',
    'lifestyle': 'Entertainment',
    'culture': 'Entertainment',
    'life & style': 'Entertainment',
    'film': 'Entertainment',
    'tv': 'Entertainment',
    'music': 'Entertainment',
    'gossip': 'Entertainment',

    # Opinion variations
    'opinion': 'Opinion',
    'opinion ': 'Opinion',
    'editorial': 'Opinion',
    'editorial ': 'Opinion',
    'blogs': 'Opinion',
    'blogs ': 'Opinion',
    'letters': 'Opinion',
    'letters ': 'Opinion',

    # Politics
    'politics': 'Politics',
    'politics ': 'Politics',
    'political': 'Politics',
    'political ': 'Politics',

    # Health
    'health': 'Health',
    'health ': 'Health',
    'healthcare': 'Health',
    'medicine': 'Health',

    # Education
    'education': 'Education',
    'education ': 'Education',
    'learning': 'Education',

    # Science
    'science': 'Science',
    'science ': 'Science',
    'research': 'Science',
}

# Visualization settings
VIZ_STYLE = "whitegrid"
VIZ_DPI = 300
VIZ_FIGSIZE = (14, 8)
VIZ_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

# Encoding settings
CSV_ENCODING = 'utf-8-sig'
SOURCE_ENCODING = 'latin-1'
