import pandas as pd
from bertopic import BERTopic
from pathlib import Path

# --- Configuration ---
PROCESSED_DATA_PATH = Path("combined_news_data_cleaned.csv")
TOPIC_MODELING_OUTPUT_DIR = Path("outputs/topic_modeling")
TOPIC_MODELING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Part 1: Data Loading and Preparation ---
print("--- Part 1: Data Loading and Preparation ---")

if not PROCESSED_DATA_PATH.exists():
    print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}.")
    print("Please run the main pipeline (python pipeline.py) first to generate the cleaned data.")
    exit()

df_clean = pd.read_csv(PROCESSED_DATA_PATH)

# Filter out articles with missing or empty descriptions for topic modeling
original_count = len(df_clean)
df_topics = df_clean.dropna(subset=['description']).copy()
df_topics = df_topics[df_topics['description'].str.strip() != ''].copy()
filtered_count = len(df_topics)

print(f"Loaded {original_count:,} articles from {PROCESSED_DATA_PATH}")
print(f"Filtered to {filtered_count:,} articles with valid descriptions for topic modeling ({original_count - filtered_count:,} removed).")

# Prepare documents list for BERTopic
documents = df_topics['description'].tolist()

print("\n--- Data Preparation Complete ---")

# --- Part 2: Model Training ---
print("\n--- Part 2: Model Training ---")
print("Initializing BERTopic model. This may download a SentenceTransformer model if not already cached.")

# You can customize the BERTopic model parameters here
# For example, to speed up for initial exploration, you might consider:
# nr_topics='auto' or a specific number, calculate_probabilities=False
# For production use, defaults are often good but slower.

topic_model = BERTopic(verbose=True, calculate_probabilities=False) # calculate_probabilities=True for advanced analysis but adds overhead

print("Training BERTopic model on documents...")
topics, probs = topic_model.fit_transform(documents)

print("BERTopic Model Training Complete.")
print(f"Discovered {len(topic_model.get_topic_info()) - 1} topics (excluding -1 for outliers).")

# Save the model for future use
model_path = TOPIC_MODELING_OUTPUT_DIR / "bertopic_model"
topic_model.save(str(model_path), serialization="safetensors")
print(f"BERTopic model saved to {model_path}")

# --- Part 3: Exploring the Results ---
print("\n--- Part 3: Exploring the Results ---")
print("Top 10 topics (excluding outliers):")
topic_info = topic_model.get_topic_info()
print(topic_info.head(11)) # +1 to exclude the outlier topic -1

# Example: Get words for a specific topic (e.g., Topic 0)
# Make sure Topic 0 exists and is not the outlier topic -1
if 0 in topic_info["Topic"].values:
    print("\nWords for Topic 0:")
    print(topic_model.get_topic(0))
else:
    print("\nTopic 0 does not exist or is an outlier. Cannot display words.")

# Assign topics back to the DataFrame
df_topics['topic'] = topics
df_topics['topic_name'] = df_topics['topic'].apply(lambda t: topic_info[topic_info['Topic'] == t]['Name'].iloc[0] if t != -1 else "Outlier Topic")

print("\n--- Topic Exploration Complete ---")

# --- Part 4: Visualization ---
print("\n--- Part 4: Visualization ---")

# Visualize Topics (2D scatter plot)
print("Generating Topic Visualization (2D scatter plot)...")
fig_topics = topic_model.visualize_topics()
if fig_topics:
    fig_topics.write_html(TOPIC_MODELING_OUTPUT_DIR / "bertopic_topics_2d.html")
    print(f"  ✓ Saved: {TOPIC_MODELING_OUTPUT_DIR / 'bertopic_topics_2d.html'}")
else:
    print("  Could not generate 2D topic visualization.")

# Visualize Top Words per Topic (Bar Chart)
print("Generating Top Words per Topic (Bar Chart)...")
fig_barchart = topic_model.visualize_barchart(top_n_topics=10) # Show top 10 topics
if fig_barchart:
    fig_barchart.write_html(TOPIC_MODELING_OUTPUT_DIR / "bertopic_top_words_barchart.html")
    print(f"  ✓ Saved: {TOPIC_MODELING_OUTPUT_DIR / 'bertopic_top_words_barchart.html'}")
else:
    print("  Could not generate top words barchart visualization.")

# Visualize Topic Hierarchy (Dendrogram)
print("Generating Topic Hierarchy (Dendrogram)...")
# Note: Hierarchical topic modeling is optional and can be computationally intensive
# Only run if you explicitly want to see the hierarchy
# fig_hierarchy = topic_model.visualize_hierarchy(top_n_topics=20)
# if fig_hierarchy:
#     fig_hierarchy.write_html(TOPIC_MODELING_OUTPUT_DIR / "bertopic_hierarchy.html")
#     print(f"  ✓ Saved: {TOPIC_MODELING_OUTPUT_DIR / 'bertopic_hierarchy.html'}")
# else:
#     print("  Could not generate topic hierarchy visualization.")


print("\n--- Visualization Complete ---")
print(f"\nAll BERTopic outputs saved to: {TOPIC_MODELING_OUTPUT_DIR}")