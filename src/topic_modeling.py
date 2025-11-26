"""
Component for Topic Modeling
"""
import pandas as pd
from bertopic import BERTopic
from pathlib import Path

def model_topics(df, output_dir):
    """
    Performs topic modeling on the 'description' column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        output_dir (Path): The directory to save model and visualizations.

    Returns:
        tuple: A tuple containing the trained model and the DataFrame with topic assignments.
    """
    print("\n" + "-" * 80)
    print("RUNNING TOPIC MODELING")
    print("-" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Data Preparation ---
    df_topics = df.dropna(subset=['description']).copy()
    df_topics = df_topics[df_topics['description'].str.strip() != ''].copy()
    documents = df_topics['description'].tolist()
    print(f"  Filtered to {len(documents):,} articles with valid descriptions for topic modeling.")

    # --- Model Training ---
    print("  Initializing and training BERTopic model (this can be very slow)...")
    topic_model = BERTopic(verbose=True, calculate_probabilities=False)
    topics, _ = topic_model.fit_transform(documents)
    print("  ✓ BERTopic Model Training Complete.")

    # --- Save Model ---
    model_path = output_dir / "bertopic_model"
    topic_model.save(str(model_path), serialization="safetensors")
    print(f"  BERTopic model saved to {model_path}")

    # --- Process Results ---
    topic_info = topic_model.get_topic_info()
    df_topics['topic'] = topics
    df_topics['topic_name'] = df_topics['topic'].apply(
        lambda t: topic_info[topic_info['Topic'] == t]['Name'].iloc[0] if t != -1 else "Outlier Topic"
    )
    print("  Assigned topic names back to DataFrame.")

    # --- Generate and Save Visualizations ---
    print("  Generating and saving visualizations...")
    fig_topics = topic_model.visualize_topics()
    if fig_topics:
        fig_topics.write_html(output_dir / "bertopic_topics_2d.html")

    fig_barchart = topic_model.visualize_barchart(top_n_topics=15)
    if fig_barchart:
        fig_barchart.write_html(output_dir / "bertopic_top_words_barchart.html")
    
    print("  ✓ Visualizations saved.")
    
    return topic_model, df_topics
