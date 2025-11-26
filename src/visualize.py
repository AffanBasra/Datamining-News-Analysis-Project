"""
Stage 3: Visualization - Generate analysis charts
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
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


def visualize_data(df):
    """
    Generate all visualizations

    Args:
        df: Cleaned DataFrame with category_clean and sentiment_polarity columns
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

    print("\n✓ Visualization complete")


if __name__ == "__main__":
    from src.ingest import ingest_data
    from src.preprocess import preprocess_data

    df = ingest_data()
    df_clean = preprocess_data(df)
    visualize_data(df_clean)

