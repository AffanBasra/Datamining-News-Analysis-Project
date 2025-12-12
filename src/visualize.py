"""
Stage 3: Visualization - Generate analysis charts
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
import ast
from src.config import TOP_CATEGORIES, FIGURES_DIR

sys.stdout.reconfigure(encoding='utf-8')
sns.set_style("whitegrid")


def prepare_viz_data(df):
    """Prepare data for visualization"""
    # Filter to top categories
    df_top = df[df['category_clean'].isin(TOP_CATEGORIES)].copy()

    # Calculate story length
    df_top['story_length'] = df_top['description'].fillna('').apply(lambda x: len(str(x).split()))

    # Parse dates
    df_top['date_parsed'] = pd.to_datetime(df_top['date'], errors='coerce')
    df_top['year_month'] = df_top['date_parsed'].dt.to_period('M')
    df_top = df_top[df_top['date_parsed'].notna()].copy()

    return df_top


def create_story_length_boxplot(df_top):
    """4.3 Story Length by Category Boxplot"""
    print("\n  [1/3] Creating Story Length by Category Boxplot...")

    plt.figure(figsize=(14, 8))
    box_data = [df_top[df_top['category_clean'] == cat]['story_length'].values
                for cat in TOP_CATEGORIES]

    bp = plt.boxplot(box_data,
                     tick_labels=TOP_CATEGORIES,
                     patch_artist=True,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Story Length (word count)', fontsize=12, fontweight='bold')
    plt.title('Story Length Distribution by Category', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    stats_text = "Box: Q1-Q3 | Line: Median | Diamond: Mean | Whiskers: 1.5×IQR"
    plt.text(0.5, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = Path(FIGURES_DIR) / "story_length_boxplot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path}")
    plt.close()


def create_source_category_heatmap(df_top):
    """4.4 Source × Category Heatmap"""
    print("\n  [2/3] Creating Source × Category Heatmap...")

    top_sources = ['Dawn', 'Tribune']
    df_heatmap = df_top[df_top['source'].isin(top_sources)].copy()

    pivot = pd.crosstab(df_heatmap['source'], df_heatmap['category_clean'])
    pivot = pivot[TOP_CATEGORIES]

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot,
                annot=True,
                fmt='d',
                cmap='YlOrRd',
                cbar_kws={'label': 'Article Count'},
                linewidths=0.5,
                linecolor='gray')

    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Source', fontsize=12, fontweight='bold')
    plt.title('Article Distribution: Source × Category', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    output_path = Path(FIGURES_DIR) / "source_category_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path}")
    plt.close()


def create_temporal_trend_lineplot(df_top):
    """4.5 Temporal Trend Line Plot"""
    print("\n  [3/3] Creating Temporal Trend Line Plot...")

    temporal_data = df_top.groupby(['year_month', 'category_clean']).size().reset_index(name='count')

    plt.figure(figsize=(16, 8))

    for cat in TOP_CATEGORIES:
        cat_data = temporal_data[temporal_data['category_clean'] == cat]
        cat_data_sorted = cat_data.sort_values('year_month')
        x = cat_data_sorted['year_month'].astype(str)
        y = cat_data_sorted['count']
        plt.plot(range(len(x)), y, marker='o', linewidth=2, markersize=4, label=cat, alpha=0.8)

    plt.xlabel('Month', fontsize=12, fontweight='bold')
    plt.ylabel('Article Count', fontsize=12, fontweight='bold')
    plt.title('Temporal Trend: Monthly Article Count by Category', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)

    all_months = sorted(temporal_data['year_month'].unique().astype(str))
    tick_positions = range(0, len(all_months), 3)
    tick_labels = [all_months[i] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=len(TOP_CATEGORIES), frameon=True, fontsize=10)

    plt.tight_layout()
    output_path = Path(FIGURES_DIR) / "temporal_trend_lineplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path}")
    plt.close()

def create_sentiment_distribution_plot(df_top):
    """New: Sentiment Polarity Distribution"""
    print("\n  [4/5] Creating Sentiment Polarity Distribution Plot...")

    plt.figure(figsize=(10, 6))
    sns.histplot(df_top['sentiment_polarity'], kde=True, bins=50, color='skyblue')
    plt.title('Distribution of Sentiment Polarity', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Sentiment Polarity (-1.0: Negative, 1.0: Positive)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Articles', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    output_path = Path(FIGURES_DIR) / "sentiment_polarity_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path}")
    plt.close()

def create_average_sentiment_by_category_source(df_top):
    """New: Average Sentiment Polarity by Category and Source"""
    print("\n  [5/5] Creating Average Sentiment by Category and Source Plots...")

    # Average sentiment by category
    avg_sentiment_cat = df_top.groupby('category_clean')['sentiment_polarity'].mean().sort_values()
    plt.figure(figsize=(12, 7))
    sns.barplot(x=avg_sentiment_cat.index, y=avg_sentiment_cat.values, palette='viridis')
    plt.title('Average Sentiment Polarity by Category', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Average Sentiment Polarity', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_path_cat = Path(FIGURES_DIR) / "avg_sentiment_by_category.png"
    plt.savefig(output_path_cat, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path_cat}")
    plt.close()

    # Average sentiment by source
    avg_sentiment_source = df_top.groupby('source')['sentiment_polarity'].mean().sort_values()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=avg_sentiment_source.index, y=avg_sentiment_source.values, palette='magma')
    plt.title('Average Sentiment Polarity by Source', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Source', fontsize=12, fontweight='bold')
    plt.ylabel('Average Sentiment Polarity', fontsize=12, fontweight='bold')
    plt.tight_layout()
    output_path_source = Path(FIGURES_DIR) / "avg_sentiment_by_source.png"
    plt.savefig(output_path_source, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path_source}")
    plt.close()


def create_ner_label_barchart(df):
    """Create a bar chart of NER label frequencies."""
    print("\n  [6/7] Creating NER Label Frequency Barchart...")
    try:
        # The 'entities' column may be a string representation of a list
        if isinstance(df['entities'].iloc[0], str):
            all_entities = df['entities'].apply(ast.literal_eval).sum()
        else:
            all_entities = df['entities'].sum()
        
        if not all_entities:
            print("    ! No entities found. Skipping NER barchart.")
            return

        labels = [ent[3] for ent in all_entities]
        label_counts = Counter(labels)
        
        if not label_counts:
            print("    ! No entity labels found. Skipping NER barchart.")
            return

        plt.figure(figsize=(12, 7))
        sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), palette='mako')
        plt.title('Frequency of Named Entity Types', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Entity Type', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_path = Path(FIGURES_DIR) / "ner_label_barchart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {output_path}")
        plt.close()

    except Exception as e:
        print(f"    ! Error creating NER barchart: {e}")


def create_ner_wordcloud(df):
    """Create a word cloud of the most frequent entities."""
    print("\n  [7/7] Creating NER Word Cloud...")
    try:
        # The 'entities' column may be a string representation of a list
        if isinstance(df['entities'].iloc[0], str):
            all_entities = df['entities'].apply(ast.literal_eval).sum()
        else:
            all_entities = df['entities'].sum()

        if not all_entities:
            print("    ! No entities found. Skipping NER wordcloud.")
            return

        entity_texts = [ent[0] for ent in all_entities if ent[3] in ['PERSON', 'ORG', 'GPE']]

        if not entity_texts:
            print("    ! No PERSON, ORG, or GPE entities found. Skipping NER wordcloud.")
            return

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(entity_texts))

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Named Entities (PERSON, ORG, GPE)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        output_path = Path(FIGURES_DIR) / "ner_wordcloud.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {output_path}")
        plt.close()

    except Exception as e:
        print(f"    ! Error creating NER wordcloud: {e}")


def create_word2vec_topic_distribution(df):
    """Create visualization for Word2Vec topic distribution."""
    # Check for both possible column names (CBOW and Skip-gram)
    topic_col = None
    model_type = ''
    if 'word2vec_topic' in df.columns:
        topic_col = 'word2vec_topic'
        model_type = 'CBOW'
    elif 'word2vec_skipgram_topic' in df.columns:
        topic_col = 'word2vec_skipgram_topic'
        model_type = 'Skip-gram'

    if topic_col is None:
        print("\n  [8/8] Skipping Word2Vec topic distribution visualization (column not found)...")
        return

    print(f"\n  [8/8] Creating Word2Vec {model_type} Topic Distribution Plot...")

    # Count articles per topic
    topic_counts = df[topic_col].value_counts().sort_index()

    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(len(topic_counts)), topic_counts.values, color='skyblue', edgecolor='navy', linewidth=0.5)
    plt.title(f'Distribution of Articles Across Word2Vec {model_type} Topics', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Topic ID', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Articles', fontsize=12, fontweight='bold')
    plt.xticks(range(len(topic_counts)), [f'Topic {i}' for i in topic_counts.index], rotation=45, ha='right')

    # Add count labels on bars
    for bar, count in zip(bars, topic_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                 str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = Path(FIGURES_DIR) / f"word2vec_{model_type.lower()}_topic_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_path}")
    plt.close()


def create_word2vec_topic_barcharts(df):
    """Create a single visualization with multiple subplots showing top terms for each Word2Vec topic."""
    # Try to get topic info, assuming it's saved in the outputs directory
    # For now, we'll create a simpler approach by looking for topic name columns
    topic_name_col = None
    topic_col = None
    model_type = ''

    if 'word2vec_topic_name' in df.columns:
        topic_name_col = 'word2vec_topic_name'
        topic_col = 'word2vec_topic'
        model_type = 'CBOW'
    elif 'word2vec_skipgram_topic_name' in df.columns:
        topic_name_col = 'word2vec_skipgram_topic_name'
        topic_col = 'word2vec_skipgram_topic'
        model_type = 'Skip-gram'

    if topic_col is None:
        print("\n  [9/9] Skipping Word2Vec topic barcharts visualization (column not found)...")
        return

    print(f"\n  [9/9] Creating Word2Vec {model_type} Topic Barcharts...")

    try:
        # Get unique topics
        unique_topics = sorted(df[topic_col].unique())
        n_topics = len(unique_topics)

        if n_topics == 0:
            print("    ! No topics found for visualization.")
            return

        # Determine subplot layout based on number of topics
        cols = min(2, n_topics)  # Max 2 columns
        rows = (n_topics + cols - 1) // cols  # Calculate rows needed

        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if n_topics == 1:
            axes = [axes]
        elif n_topics > 1:
            axes = axes.flatten() if n_topics > 1 else [axes]

        # Create a subplot for each topic
        for i, topic_id in enumerate(unique_topics):
            topic_df = df[df[topic_col] == topic_id]
            if len(topic_df) > 0:
                # Use the topic name to get sample terms (from topic_name column)
                topic_name = topic_df.iloc[0][topic_name_col] if topic_name_col in df.columns else f"Topic {topic_id}"
                top_terms = [term.strip() for term in topic_name.split(',')[:10]]  # Get top 10 terms

                if len(top_terms) > 0:
                    # Create a horizontal bar chart for this topic
                    y_pos = range(len(top_terms))
                    # Using dummy values as we don't have actual term frequencies
                    values = list(range(len(top_terms), 0, -1))  # Decreasing values for better visualization
                    bars = axes[i].barh(y_pos, values, color='lightblue', edgecolor='navy')
                    axes[i].set_yticks(y_pos)
                    axes[i].set_yticklabels(top_terms)
                    axes[i].set_xlabel('Term Importance')
                    axes[i].set_title(f'Topic {topic_id}: {", ".join(top_terms[:3])}...', fontsize=10)

                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        axes[i].text(bar.get_width() * 0.5, bar.get_y() + bar.get_height()/2,
                                   str(value), ha='center', va='center', fontsize=8)

        # Hide unused subplots
        for i in range(n_topics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Top Terms in Word2Vec {model_type} Topics', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = Path(FIGURES_DIR) / f"word2vec_{model_type.lower()}_all_topics_barchart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {output_path}")

    except Exception as e:
        print(f"    ! Error creating topic barcharts: {e}")


def create_word2vec_wordclouds(df):
    """Create a single visualization with word clouds for each Word2Vec topic."""
    topic_name_col = None
    topic_col = None
    model_type = ''

    if 'word2vec_topic_name' in df.columns:
        topic_name_col = 'word2vec_topic_name'
        topic_col = 'word2vec_topic'
        model_type = 'CBOW'
    elif 'word2vec_skipgram_topic_name' in df.columns:
        topic_name_col = 'word2vec_skipgram_topic_name'
        topic_col = 'word2vec_skipgram_topic'
        model_type = 'Skip-gram'

    if topic_col is None:
        print("\n  [10/10] Skipping Word2Vec word clouds visualization (column not found)...")
        return

    print(f"\n  [10/10] Creating Word2Vec {model_type} Topic Word Clouds...")

    try:
        # Get unique topics
        unique_topics = sorted(df[topic_col].unique())
        n_topics = len(unique_topics)

        if n_topics == 0:
            print("    ! No topics found for visualization.")
            return

        # Determine subplot layout based on number of topics
        cols = min(2, n_topics)  # Max 2 columns
        rows = (n_topics + cols - 1) // cols  # Calculate rows needed

        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
        if n_topics == 1:
            axes = [axes]
        elif n_topics > 1:
            axes = axes.flatten() if n_topics > 1 else [axes]

        # Create a word cloud subplot for each topic
        for i, topic_id in enumerate(unique_topics):
            topic_df = df[df[topic_col] == topic_id]
            if len(topic_df) > 0:
                # Use the topic name to get sample terms
                topic_name = topic_df.iloc[0][topic_name_col] if topic_name_col in df.columns else f"Topic {topic_id}"
                top_terms = [term.strip() for term in topic_name.split(',')[:15]]  # Get top 15 terms

                if len(top_terms) > 0:
                    # Create text for the word cloud (repeat terms to show importance)
                    text = ' '.join(top_terms * 3)  # Repeat to make more visible

                    # Create word cloud
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='viridis',
                        max_words=20,
                        relative_scaling=0.5
                    ).generate(text)

                    # Display word cloud in subplot
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].axis('off')
                    axes[i].set_title(f'Topic {topic_id}: {", ".join(top_terms[:3])}...', fontsize=12, fontweight='bold')

        # Hide unused subplots
        for i in range(n_topics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Word2Vec {model_type} Topic Word Clouds', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = Path(FIGURES_DIR) / f"word2vec_{model_type.lower()}_all_topics_wordcloud.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {output_path}")

    except Exception as e:
        print(f"    ! Error creating topic word clouds: {e}")


def visualize_data(df, run_ner=False):
    """
    Generate all visualizations

    Args:
        df: Cleaned DataFrame with category_clean and sentiment_polarity columns
        run_ner (bool): Whether NER analysis was run
    """
    print("\n" + "=" * 80)
    print("STAGE 3: VISUALIZATION")
    print("=" * 80)

    print("\n[1/2] Preparing data for visualization...")
    df_top = prepare_viz_data(df)
    print(f"  Filtered to top {len(TOP_CATEGORIES)} categories: {len(df_top):,} articles")
    print(f"  Date range: {df_top['date_parsed'].min().date()} to {df_top['date_parsed'].max().date()}")

    print("\n[2/2] Creating visualizations...")
    create_story_length_boxplot(df_top)
    create_source_category_heatmap(df_top)
    create_temporal_trend_lineplot(df_top)
    create_sentiment_distribution_plot(df_top)
    create_average_sentiment_by_category_source(df_top)

    if run_ner and 'entities' in df.columns:
        print("\n[NER Visualizations] Creating NER visualizations...")
        create_ner_label_barchart(df)
        create_ner_wordcloud(df)
    elif run_ner and 'entities' not in df.columns:
        print("\n[NER Visualizations] Warning: NER was requested but entities column not found...")
    else:
        print("\n[NER Visualizations] Skipping NER visualizations (NER not requested)...")

    # Create Word2Vec topic visualization if available
    create_word2vec_topic_distribution(df)
    create_word2vec_topic_barcharts(df)
    create_word2vec_wordclouds(df)

    print("\n✓ Visualization complete")


if __name__ == "__main__":
    from src.ingest import ingest_data
    from src.preprocess import preprocess_data

    df = ingest_data()
    df_clean = preprocess_data(df)
    visualize_data(df_clean, run_ner=False)

