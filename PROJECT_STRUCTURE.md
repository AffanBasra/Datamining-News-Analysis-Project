# Project Structure

## Overview
This project follows a clean, modular pipeline architecture for news data analysis.

## Directory Structure

```
DataMiningProject/
├── pipeline.py                         # Main pipeline orchestrator
├── main.py                            # Quick entry point
├── requirements.txt                   # Python dependencies
├── README.md                          # Full documentation
├── PROJECT_STRUCTURE.md               # This file
│
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── config.py                      # Configuration settings
│   ├── ingest.py                      # Stage 1: Data ingestion
│   ├── preprocess.py                  # Stage 2: Preprocessing
│   └── visualize.py                   # Stage 3: Visualization
│
├── data/                              # Data directories
│   ├── raw/                           # Raw data snapshots (empty - sources external)
│   └── processed/                     # Processed data (empty - outputs at root)
│
├── outputs/                           # Pipeline outputs
│   ├── figures/                       # Generated visualizations
│   │   ├── story_length_boxplot.png
│   │   ├── source_category_heatmap.png
│   │   └── temporal_trend_lineplot.png
│   └── reports/                       # Analysis reports (future)
│
├── experiments/                       # Ad-hoc analysis scripts
│   ├── analyze_data.py
│   ├── clean_categories.py
│   ├── complete_analysis.py
│   ├── data_preprocessing.py
│   └── visualizations.py
│
├── tests/                             # Unit tests (future)
│
└── combined_news_data_cleaned.csv     # Final processed dataset
```

## Pipeline Stages

### Stage 1: Ingestion (`src/ingest.py`)
**Purpose**: Load and combine raw data
- Reads Dawn CSV (45,068 articles)
- Reads Tribune CSV (93,596 articles)
- Combines into single DataFrame (138,664 articles)
- Removes unnecessary columns

**Input**: External CSV files
**Output**: Raw combined DataFrame

### Stage 2: Preprocessing (`src/preprocess.py`)
**Purpose**: Clean and normalize data
- Category normalization (558 → 53 categories)
- Apply mapping rules (Sport→Sports, Tech→Technology, etc.)
- Handle multi-categories and parsing errors
- Add `category_clean` column

**Input**: Raw DataFrame from Stage 1
**Output**: Cleaned DataFrame

### Stage 3: Visualization (`src/visualize.py`)
**Purpose**: Generate analysis charts
- Filter to top 6 categories
- Calculate story length (word count)
- Parse dates for temporal analysis
- Generate 3 visualizations

**Input**: Cleaned DataFrame from Stage 2
**Output**: PNG files in `outputs/figures/`

## Usage

### Run Complete Pipeline
```bash
python pipeline.py
```
Executes all 3 stages and saves:
- `combined_news_data_cleaned.csv`
- All visualizations in `outputs/figures/`

### Quick Data Overview
```bash
python main.py
```
Loads existing cleaned data and shows summary statistics.

### Run Individual Stages
```bash
python -m src.ingest       # Stage 1 only
python -m src.preprocess   # Stages 1-2
python -m src.visualize    # Stages 1-3
```

## Configuration

All settings are centralized in `src/config.py`:
- Data source paths
- Top categories for analysis
- Output directories
- Visualization settings

## Data Flow

```
External CSVs
    ↓
[Stage 1: Ingest]
    ↓
Raw Combined DataFrame (138,664 rows)
    ↓
[Stage 2: Preprocess]
    ↓
Cleaned DataFrame (category_clean added)
    ↓
[Stage 3: Visualize]
    ↓
Filtered DataFrame (top 6 categories)
    ↓
3 Visualization PNGs
```

## Key Features

1. **Modular Design**: Each stage is independent and can run standalone
2. **Clean Separation**: Configuration, ingestion, preprocessing, and visualization separated
3. **Minimal Coupling**: Modules only import what they need
4. **Easy Extension**: Add new stages or modify existing ones without affecting others
5. **Experiments Folder**: Old/ad-hoc scripts preserved but separated

## Future Extensions

This structure supports adding:
- `src/features/` - Feature engineering (TF-IDF, embeddings, etc.)
- `src/models/` - Machine learning models
- `src/validation/` - Data validation checks
- `tests/` - Unit and integration tests
- More visualization types in `src/visualize.py`
