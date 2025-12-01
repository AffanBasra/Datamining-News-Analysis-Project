# News-Lens: Automated News Analysis Pipeline

## Project Overview

News-Lens is a modular and extensible data pipeline designed for in-depth analysis of news articles. Initially developed for Pakistani news sources (Dawn and Tribune), it aims to uncover hidden insights by processing raw news data through various stages, including advanced natural language processing techniques like sentiment analysis, topic modeling, and named entity recognition.

This tool is built with a journalist's perspective in mind, allowing the analysis of new, unseen news sources to understand their editorial focus, sentiment, key themes, and mentioned entities, potentially revealing biases or unique coverages compared to a baseline.

## Features

*   **Data Ingestion:** Loads news articles from specified CSV files (Dawn and Tribune).
*   **Base Preprocessing:** Cleans and normalizes article categories for consistent analysis.
*   **Sentiment Analysis:** Utilizes state-of-the-art Flair models to determine the emotional tone (polarity) of news descriptions. Highly optimized for GPU usage.
*   **Topic Modeling:** Employs BERTopic to automatically discover and visualize latent topics and themes within the news content.
*   **Named Entity Recognition (NER):** Uses spaCy to identify and extract named entities (e.g., people, organizations, locations) from the news articles.
*   **Intelligent Caching:** Skips time-consuming reprocessing steps if intermediate results are already available, significantly speeding up subsequent runs.
*   **Modular Design:** Each stage (ingestion, preprocessing, sentiment, topic modeling, NER) is a separate, reusable component.
*   **Granular Control:** Offers fine-grained control over which analysis steps to run via command-line flags.
*   **Comprehensive Visualizations:** Generates various charts and interactive reports for key insights into story length, source/category distribution, temporal trends, sentiment, discovered topics, and named entities.

## Project Structure

```
.
├── main.py                     # Entry point for the main pipeline (deprecated)
├── pipeline.py                 # Primary orchestrator for the modular analysis pipeline
├── PROJECT_STRUCTURE.md        # High-level project architecture documentation
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .git/                       # Git version control
├── DMProject/                  # Python virtual environment
├── experiments/                # Scripts for ad-hoc analysis and testing
│   ├── analyze_data.py
│   ├── clean_categories.py
│   ├── complete_analysis.py
│   ├── data_preprocessing.py
│   └── visualizations.py
│   └── topic_modeling.py       # (Legacy) Topic modeling script, now integrated via src/topic_modeling.py
├── outputs/
│   ├── figures/                # Stores generated PNG visualization charts
│   └── topic_modeling/         # Stores BERTopic models and interactive HTML visualizations
├── src/                        # Core source code modules
│   ├── __init__.py
│   ├── config.py               # Configuration settings (data paths, top categories)
│   ├── ingest.py               # Stage 1: Data ingestion (loads raw CSVs)
│   ├── preprocess.py           # Stage 2: Base preprocessing (category cleaning)
│   ├── sentiment_analysis.py   # Module for Flair-based sentiment analysis
│   ├── topic_modeling.py       # Module for BERTopic-based topic modeling
│   ├── ner.py                  # Module for spaCy-based Named Entity Recognition
│   └── visualize.py            # Stage 3: Visualization generation
└── data/                       # Directory for raw data files
    └── raw/                    # Subdirectory for raw input CSVs (Dawn and Tribune)
        ├── dawn.csv
        └── tribune.csv
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_repo/your_project.git
    cd your_project
    ```

2.  **Prepare your data:**
    Place your `dawn.csv` and `tribune.csv` files into the `data/raw/` directory.

3.  **Install Python Dependencies (Crucial for GPU Acceleration):**
    First, ensure you have a compatible Python version (3.9+ recommended).
    
    **A. Install PyTorch with CUDA (Highly Recommended for Performance):**
    `flair` and `BERTopic` leverage PyTorch. For significant speedup, especially for sentiment analysis and topic modeling, install the GPU-enabled version of PyTorch if you have an NVIDIA GPU.
    *   **Uninstall any existing PyTorch (CPU-only) installation first:**
        ```bash
        pip uninstall torch
        ```
    *   **Visit the official PyTorch website:** Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   **Select your system configuration:** Choose your OS, Package (pip), Language (Python), and most importantly, your CUDA version (e.g., CUDA 11.8, CUDA 12.1).
    *   **Run the generated installation command:** The website will provide a specific `pip install` command (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`). Execute this command.

    **B. Install remaining dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install `flair`, `bertopic`, `spacy`, pandas, and other libraries.

## How to Run the Pipeline

The `pipeline.py` script is the main entry point and offers flexible execution options.

**Basic Usage:**

*   **Run with default settings (loads from cache, runs sentiment if missing, skips topics and NER):**
    ```bash
    python pipeline.py
    ```

*   **Run on a smaller sample (e.g., 5000 articles) for faster testing/development:**
    ```bash
    python pipeline.py --sample_size 5000
    ```

**Advanced Usage with Flags:**

*   **Force re-running base data ingestion and category cleaning (and all subsequent steps):**
    ```bash
    python pipeline.py --force_reprocess_base
    ```

*   **Force re-running sentiment analysis (even if columns exist) on the full dataset:**
    ```bash
    python pipeline.py --force_reprocess_sentiment
    ```

*   **Run topic modeling (this is skipped by default, can be slow):**
    ```bash
    python pipeline.py --run_topic_modeling
    ```

*   **Run Named Entity Recognition (this is also skipped by default):**
    ```bash
    python pipeline.py --run_ner
    ```

*   **Combine flags (e.g., re-run sentiment, run topics, and run NER on a sample):**
    ```bash
    python pipeline.py --sample_size 10000 --force_reprocess_sentiment --run_topic_modeling --run_ner
    ```

*   **To run everything from scratch on a full dataset:**
    ```bash
    python pipeline.py --force_reprocess_base --force_reprocess_sentiment --run_topic_modeling --run_ner
    ```

## Outputs

The pipeline generates the following:

*   **Processed Data:** A CSV file (`combined_news_data_cleaned.csv` or `combined_news_data_cleaned_sample_XXXX.csv`) in the root directory containing the enriched dataset with cleaned categories, sentiment polarity, topic assignments, and named entities (if run).
*   **Visualization Figures:** PNG images in the `outputs/figures/` directory, including:
    *   `story_length_boxplot.png`
    *   `source_category_heatmap.png`
    *   `temporal_trend_lineplot.png`
    *   `sentiment_polarity_distribution.png`
    *   `avg_sentiment_by_category.png`
    *   `avg_sentiment_by_source.png`
    *   `ner_label_barchart.png`
    *   `ner_wordcloud.png`
*   **Topic Modeling Reports:** Interactive HTML files and the trained BERTopic model in the `outputs/topic_modeling/` directory, including:
    *   `bertopic_model` (saved BERTopic model)
    *   `bertopic_topics_2d.html` (interactive 2D topic map)
    *   `bertopic_top_words_barchart.html` (interactive bar charts of top words per topic)

## Key Insights (Example, from Initial Analysis)

This section would typically contain summaries of key findings from your analysis.
*(Example from previous README, to be updated with insights from sentiment/topics/NER)*

*   **Content Focus**: Pakistan-related news often dominates both sources.
*   **Story Length Patterns**: Political/domestic topics tend to have longer stories, while technology might have shorter, more concise ones.
*   **Source Differences**: One source might focus more on serious news, while another offers broader coverage including entertainment and technology.
*   **Temporal Trends**: Observe how article counts for different categories change over time, potentially correlating with real-world events.
*   **Sentiment Trends**: How does the sentiment of news change over time, or differ between sources/categories?
*   **Discovered Topics**: What are the actual underlying themes and sub-topics discussed in the news, as revealed by BERTopic?
*   **Key Entities**: What are the most frequently mentioned people, organizations, and locations in the news?