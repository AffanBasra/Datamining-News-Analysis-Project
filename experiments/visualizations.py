import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Load the cleaned dataset
print("Loading dataset...")
df = pd.read_csv(
    "C:\\Users\\DELL\\Desktop\\Data Mining\\DataMiningProject\\combined_news_data_cleaned.csv",
    encoding='utf-8-sig'
)

print(f"Dataset loaded: {df.shape}")

# Filter to top categories only
print("\nFiltering to top categories...")
top_categories = ['Pakistan', 'World', 'Sports', 'Business', 'Entertainment', 'Technology']
df_top = df[df['category_clean'].isin(top_categories)].copy()
print(f"Filtered dataset: {df_top.shape}")
print(f"Categories: {top_categories}")

# Calculate story length (word count in description)
print("\nCalculating story lengths...")
df_top['story_length'] = df_top['description'].fillna('').apply(lambda x: len(str(x).split()))
print(f"Story length range: {df_top['story_length'].min()} to {df_top['story_length'].max()} words")
print(f"Mean story length: {df_top['story_length'].mean():.2f} words")

# Parse dates for temporal analysis
print("\nParsing dates...")
df_top['date_parsed'] = pd.to_datetime(df_top['date'], errors='coerce')
df_top['year_month'] = df_top['date_parsed'].dt.to_period('M')

# Remove rows with invalid dates
df_top = df_top[df_top['date_parsed'].notna()].copy()
print(f"Dataset after date parsing: {df_top.shape}")
print(f"Date range: {df_top['date_parsed'].min()} to {df_top['date_parsed'].max()}")

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# 4.3 Story Length by Category Boxplot
# ============================================================================
print("\n[1/3] Creating Story Length by Category Boxplot...")

plt.figure(figsize=(14, 8))
box_data = [df_top[df_top['category_clean'] == cat]['story_length'].values
            for cat in top_categories]

bp = plt.boxplot(box_data,
                 labels=top_categories,
                 patch_artist=True,
                 showmeans=True,
                 meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

# Color the boxes
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.xlabel('Category', fontsize=12, fontweight='bold')
plt.ylabel('Story Length (word count)', fontsize=12, fontweight='bold')
plt.title('Story Length Distribution by Category', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add statistics annotation
stats_text = "Box: Q1-Q3 | Line: Median | Diamond: Mean | Whiskers: 1.5×IQR"
plt.text(0.5, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=9, verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('C:\\Users\\DELL\\Desktop\\Data Mining\\DataMiningProject\\story_length_boxplot.png',
            dpi=300, bbox_inches='tight')
print("  ✓ Saved: story_length_boxplot.png")

# Print statistics
print("\n  Story Length Statistics by Category:")
for cat in top_categories:
    cat_data = df_top[df_top['category_clean'] == cat]['story_length']
    print(f"    {cat:15} - Median: {cat_data.median():6.1f}, Mean: {cat_data.mean():6.1f}, Std: {cat_data.std():6.1f}")

# ============================================================================
# 4.4 Source × Category Heatmap
# ============================================================================
print("\n[2/3] Creating Source × Category Heatmap...")

# Create crosstab (only for top sources)
top_sources = ['Dawn', 'Tribune']
df_heatmap = df_top[df_top['source'].isin(top_sources)].copy()

pivot = pd.crosstab(df_heatmap['source'], df_heatmap['category_clean'])
# Reorder columns to match top_categories
pivot = pivot[top_categories]

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
plt.savefig('C:\\Users\\DELL\\Desktop\\Data Mining\\DataMiningProject\\source_category_heatmap.png',
            dpi=300, bbox_inches='tight')
print("  ✓ Saved: source_category_heatmap.png")

# Print crosstab
print("\n  Source × Category Crosstab:")
print(pivot.to_string())

# Calculate percentages
print("\n  Percentage by Source:")
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
print(pivot_pct.round(2).to_string())

# ============================================================================
# 4.5 Temporal Trend Line Plot
# ============================================================================
print("\n[3/3] Creating Temporal Trend Line Plot...")

# Group by month and category
temporal_data = df_top.groupby(['year_month', 'category_clean']).size().reset_index(name='count')

plt.figure(figsize=(16, 8))

# Plot line for each category
for cat in top_categories:
    cat_data = temporal_data[temporal_data['category_clean'] == cat]
    # Convert period to timestamp for plotting
    cat_data_sorted = cat_data.sort_values('year_month')
    x = cat_data_sorted['year_month'].astype(str)
    y = cat_data_sorted['count']

    plt.plot(range(len(x)), y, marker='o', linewidth=2, markersize=4, label=cat, alpha=0.8)

plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Article Count', fontsize=12, fontweight='bold')
plt.title('Temporal Trend: Monthly Article Count by Category', fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)

# Set x-axis labels (show every 3rd month to avoid crowding)
all_months = sorted(temporal_data['year_month'].unique().astype(str))
tick_positions = range(0, len(all_months), 3)
tick_labels = [all_months[i] for i in tick_positions]
plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')

# Place legend below the chart
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=len(top_categories), frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('C:\\Users\\DELL\\Desktop\\Data Mining\\DataMiningProject\\temporal_trend_lineplot.png',
            dpi=300, bbox_inches='tight')
print("  ✓ Saved: temporal_trend_lineplot.png")

# Print temporal statistics
print("\n  Temporal Statistics:")
print(f"    Date range: {df_top['date_parsed'].min().strftime('%Y-%m-%d')} to {df_top['date_parsed'].max().strftime('%Y-%m-%d')}")
print(f"    Total months: {len(all_months)}")
print(f"\n  Average monthly article count by category:")
monthly_avg = temporal_data.groupby('category_clean')['count'].mean().sort_values(ascending=False)
for cat, avg in monthly_avg.items():
    print(f"    {cat:15} {avg:6.1f} articles/month")

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("=" * 80)
print("\nFiles saved:")
print("  1. story_length_boxplot.png")
print("  2. source_category_heatmap.png")
print("  3. temporal_trend_lineplot.png")
