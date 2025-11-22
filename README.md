# News Dataset Analysis - Data Mining Project

## Project Overview
This project analyzes news articles from two major Pakistani news sources (Dawn and Tribune) spanning from January 2020 to July 2023.

## Project Structure
```
DataMiningProject/
├── pipeline.py              # Main pipeline orchestrator
├── src/
│   ├── config.py           # Configuration settings
│   ├── ingest.py           # Stage 1: Data ingestion
│   ├── preprocess.py       # Stage 2: Preprocessing & normalization
│   └── visualize.py        # Stage 3: Visualization
├── outputs/
│   └── figures/            # Generated visualizations
├── experiments/            # Ad-hoc analysis scripts
├── combined_news_data_cleaned.csv  # Processed dataset
└── README.md
```

## Dataset Summary
- **Total Articles**: 138,664 news articles
- **Sources**:
  - Tribune: 93,596 articles
  - Dawn: 45,068 articles
- **Date Range**: January 1, 2020 - July 19, 2023
- **Duration**: 43 months

## Data Preprocessing

### 1. Data Concatenation
- Combined Dawn and Tribune datasets into a single dataframe
- Removed 91 empty "Unnamed" columns from Dawn dataset
- Standardized column structure across both sources

### 2. Category Normalization
Applied comprehensive category mapping to reduce **558 → 53 unique categories**:

**Key Mappings:**
- `Sport` / `sport` / `sports` → `Sports`
- `Tech` → `Technology`
- `Film` / `TV` / `Music` / `Gossip` / `Life & Style` → `Entertainment`
- Multi-categories (comma-separated) → First category only
- Long text (>50 chars, parsing errors) → `Unknown`

### 3. Top 6 Categories Selected
For detailed analysis, we focused on the top 6 categories:
1. **Pakistan** - 40,991 articles (34.30%)
2. **World** - 27,518 articles (23.03%)
3. **Sports** - 19,601 articles (16.40%)
4. **Business** - 16,468 articles (13.78%)
5. **Entertainment** - 9,081 articles (7.60%)
6. **Technology** - 5,834 articles (4.88%)

## Visualizations Created

### 4.3 Story Length by Category (Boxplot)
**File**: `story_length_boxplot.png`

**Description**:
- X-axis: Category
- Y-axis: Story length (word count)
- Shows median, quartiles, and mean (red diamond) for each category

**Key Findings**:
- **Pakistan** articles are longest: Median = 422 words, Mean = 496 words
- **Technology** articles are shortest: Median = 299 words, Mean = 356 words
- Business and Pakistan categories show similar length patterns
- Sports and World categories have tighter distributions (lower variability)

| Category | Median | Mean | Std Dev |
|----------|--------|------|---------|
| Pakistan | 422 | 496 | 352 |
| World | 372 | 427 | 242 |
| Sports | 379 | 425 | 203 |
| Business | 425 | 478 | 255 |
| Entertainment | 366 | 435 | 237 |
| Technology | 299 | 356 | 221 |

### 4.4 Source × Category Heatmap
**File**: `source_category_heatmap.png`

**Description**:
- Crosstab showing article distribution across sources and categories
- Darker colors indicate higher article counts
- Annotated with exact counts

**Key Findings**:
- Dawn has **NO** Entertainment or Technology coverage in this dataset
- Dawn focuses heavily on Pakistan news (53.34% of their articles)
- Tribune has more balanced category distribution
- Tribune dominates Sports coverage (16,065 vs 3,536)

**Distribution by Source**:

| Source | Pakistan | World | Sports | Business | Entertainment | Technology |
|--------|----------|-------|--------|----------|---------------|------------|
| Dawn | 21,127 (53.34%) | 9,788 (24.71%) | 3,536 (8.93%) | 5,156 (13.02%) | 0 (0%) | 0 (0%) |
| Tribune | 19,864 (24.87%) | 17,730 (22.19%) | 16,065 (20.11%) | 11,312 (14.16%) | 9,081 (11.37%) | 5,834 (7.30%) |

### 4.5 Temporal Trend Line Plot
**File**: `temporal_trend_lineplot.png`

**Description**:
- Monthly article counts for each category over time
- X-axis: Month (Jan 2020 - Jul 2023)
- Y-axis: Article count
- Legend placed below the chart

**Key Findings**:
- Pakistan news maintains highest volume throughout (avg 953 articles/month)
- Significant spike in Pakistan coverage during certain political events
- Sports coverage shows seasonal variations
- Technology coverage remains relatively stable and low
- Entertainment shows steady moderate coverage (avg 211 articles/month)

**Average Monthly Article Count**:
- Pakistan: 953.3 articles/month
- World: 640.0 articles/month
- Sports: 455.8 articles/month
- Business: 383.0 articles/month
- Entertainment: 211.2 articles/month
- Technology: 135.7 articles/month

## Key Insights

1. **Content Focus**: Pakistan-related news dominates both sources, reflecting the domestic focus of these news outlets

2. **Story Length Patterns**:
   - Longer stories for political/domestic topics (Pakistan, Business)
   - Shorter, more concise stories for Technology
   - Relatively consistent length for Sports articles

3. **Source Differences**:
   - Dawn: More focused on serious news (Pakistan, World, Business)
   - Tribune: Broader coverage including Entertainment and Technology

4. **Temporal Trends**:
   - Consistent high volume of Pakistan news throughout the period
   - Sports coverage shows variability (likely tied to major sporting events)
   - Relatively stable patterns for other categories

5. **Data Quality**:
   - Only 4 missing descriptions out of 138,664 articles (99.997% complete)
   - 1,912 duplicate headlines (1.38%)
   - 666 duplicate links (0.48%)

## Files in This Project

### Data Files
- `combined_news_data.csv` - Initial combined dataset
- `combined_news_data_cleaned.csv` - Cleaned dataset with category normalization

### Scripts
- `main.py` - Main script to load and display cleaned data
- `data_preprocessing.py` - Initial data concatenation and preprocessing
- `clean_categories.py` - Enhanced category cleaning and normalization
- `visualizations.py` - Generate all three visualizations
- `complete_analysis.py` - Complete end-to-end analysis pipeline
- `analyze_data.py` - Initial data exploration script

### Visualizations
- `story_length_boxplot.png` - Story length distribution by category
- `source_category_heatmap.png` - Source × Category crosstab heatmap
- `temporal_trend_lineplot.png` - Monthly article count trends

## Pipeline Stages

### Stage 1: Ingestion (`src/ingest.py`)
- Loads Dawn and Tribune CSV datasets
- Removes unnecessary columns
- Combines into single DataFrame
- **Output**: Raw combined dataset (138,664 articles)

### Stage 2: Preprocessing (`src/preprocess.py`)
- Category normalization (558 → 53 categories)
- Mappings:
  - `Sport/sport/sports` → `Sports`
  - `Tech` → `Technology`
  - `Film/TV/Music/Gossip` → `Entertainment`
  - Multi-categories → First category
  - Long text (>50 chars) → `Unknown`
- **Output**: Cleaned dataset with `category_clean` column

### Stage 3: Visualization (`src/visualize.py`)
- Filters to top 6 categories
- Generates 3 visualizations:
  1. Story Length by Category (Boxplot)
  2. Source × Category Heatmap
  3. Temporal Trend Line Plot
- **Output**: PNG files in `outputs/figures/`

## How to Run

**Run the complete pipeline:**
```bash
python pipeline.py
```

This will execute all 3 stages and generate:
- `combined_news_data_cleaned.csv` - Cleaned dataset
- `outputs/figures/*.png` - All visualizations

**Run individual stages:**
```bash
python -m src.ingest       # Stage 1 only
python -m src.preprocess   # Stages 1-2
python -m src.visualize    # Stages 1-3
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib seaborn
```

## Next Steps for Analysis
- Text mining and NLP analysis on headlines/descriptions
- Sentiment analysis by category
- Topic modeling (LDA, NMF)
- Classification models to predict categories
- Named entity recognition for key figures/locations
- Word clouds by category
- N-gram analysis
