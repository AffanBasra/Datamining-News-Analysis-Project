"""
Component for Topic Modeling using Word2Vec CBOW
"""
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


def preprocess_for_word2vec(texts):
    """
    Preprocess texts for Word2Vec training

    Args:
        texts (list): List of text strings

    Returns:
        list: List of tokenized sentences
    """
    # Define common English stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when',
        'where', 'why', 'how', 'if', 'so', 'than', 'too', 'very', 'just', 'now',
        'not', 'no', 'yes', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
    }

    processed_texts = []
    for text in tqdm(texts, desc="Preprocessing texts for Word2Vec"):
        if pd.isna(text) or str(text).strip() == '':
            continue
        # Simple preprocessing: split into tokens, convert to lowercase
        tokens = str(text).lower().split()
        # Remove stop words, filter out very short tokens, and punctuation
        tokens = [
            token.strip()
            for token in tokens
            if len(token) > 2 and token not in stop_words and token.isalpha()
        ]
        if tokens:
            processed_texts.append(tokens)

    return processed_texts


def train_word2vec_model(processed_texts, vector_size=100, window=5, min_count=4, workers=4, sg=0):
    """
    Train a Word2Vec model (CBOW or Skip-gram) on the preprocessed texts

    Args:
        processed_texts (list): List of tokenized sentences
        vector_size (int): Dimensionality of the word vectors
        window (int): Maximum distance between the current and predicted word
        min_count (int): Ignores all words with total frequency lower than this
        workers (int): Number of worker threads
        sg (int): Training algorithm - 0 for CBOW, 1 for Skip-gram

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model
    """
    model_name = "CBOW" if sg == 0 else "Skip-gram"
    print(f"  Training Word2Vec {model_name} model...")
    model = Word2Vec(
        sentences=processed_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg
    )
    print(f"  ✓ Word2Vec {model_name} model trained with {len(model.wv.key_to_index)} unique words")
    return model


def get_document_vectors(df, word2vec_model, text_col='description_clean'):
    """
    Compute document vectors by averaging word vectors

    Args:
        df (pd.DataFrame): Input DataFrame
        word2vec_model: Trained Word2Vec model
        text_col (str): Name of the text column

    Returns:
        np.array: Document vectors
    """
    # Define common English stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when',
        'where', 'why', 'how', 'if', 'so', 'than', 'too', 'very', 'just', 'now',
        'not', 'no', 'yes', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
    }

    print("  Computing document vectors...")
    doc_vectors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing document vectors"):
        text = row[text_col]
        if pd.isna(text) or str(text).strip() == '':
            # Use zero vector for empty texts
            doc_vectors.append(np.zeros(word2vec_model.wv.vector_size))
            continue

        tokens = str(text).lower().split()
        # Remove stop words and filter for alpha tokens that exist in the model
        tokens = [
            token.strip()
            for token in tokens
            if len(token) > 2 and token in word2vec_model.wv.key_to_index
            and token not in stop_words and token.isalpha()
        ]

        if tokens:
            # Average the word vectors for this document
            vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv.key_to_index]
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(word2vec_model.wv.vector_size)
        else:
            doc_vector = np.zeros(word2vec_model.wv.vector_size)

        doc_vectors.append(doc_vector)

    doc_vectors = np.array(doc_vectors)
    print(f"  ✓ Computed {len(doc_vectors)} document vectors")
    return doc_vectors


def cluster_documents(doc_vectors, n_clusters=10):
    """
    Cluster documents using K-means to identify topics

    Args:
        doc_vectors (np.array): Document vectors
        n_clusters (int): Number of clusters (topics)

    Returns:
        tuple: (cluster_labels, kmeans_model)
    """
    print(f"  Clustering documents into {n_clusters} topics using K-means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    cluster_labels = kmeans.fit_predict(doc_vectors)
    print(f"  ✓ Clustering complete with {len(np.unique(cluster_labels))} clusters")
    return cluster_labels, kmeans


def extract_words_per_cluster(df, cluster_labels, word2vec_model, text_col='description_clean', top_n=10):
    """
    Extract top words per cluster to identify topics

    Args:
        df (pd.DataFrame): Input DataFrame
        cluster_labels (list): Cluster assignments for each document
        word2vec_model: Trained Word2Vec model
        text_col (str): Name of the text column
        top_n (int): Number of top words per cluster

    Returns:
        dict: Cluster to top words mapping
    """
    # Define common English stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when',
        'where', 'why', 'how', 'if', 'so', 'than', 'too', 'very', 'just', 'now',
        'not', 'no', 'yes', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
    }

    print("  Extracting top words per cluster...")
    cluster_to_words = {}

    # Group documents by cluster
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in tqdm(unique_clusters, desc="Extracting words per cluster"):
        cluster_docs_idx = np.where(cluster_labels == cluster_id)[0]
        cluster_texts = df.iloc[cluster_docs_idx][text_col].tolist()

        # Combine all texts in the cluster
        cluster_tokens = []
        for text in cluster_texts:
            if not (pd.isna(text) or str(text).strip() == ''):
                tokens = str(text).lower().split()
                # Remove stop words and filter for alpha tokens that exist in the model
                tokens = [
                    token.strip()
                    for token in tokens
                    if len(token) > 2 and token in word2vec_model.wv.key_to_index
                    and token not in stop_words and token.isalpha()
                ]
                cluster_tokens.extend(tokens)

        # Find most frequent tokens in cluster
        if cluster_tokens:
            token_counts = {}
            for token in cluster_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            # Sort by frequency
            sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
            top_words = [item[0] for item in sorted_tokens[:top_n]]
        else:
            top_words = ["unknown"]

        cluster_to_words[cluster_id] = top_words

    print("  ✓ Top words extraction complete")
    return cluster_to_words


def model_topics_word2vec(df, output_dir, n_clusters=10, vector_size=100, min_count=4, model_type='cbow'):
    """
    Performs topic modeling using Word2Vec (CBOW or Skip-gram) and K-means clustering.

    Args:
        df (pd.DataFrame): The input DataFrame.
        output_dir (Path): The directory to save model and visualizations.
        n_clusters (int): Number of clusters/topics to identify.
        vector_size (int): Dimensionality of the Word2Vec vectors.
        min_count (int): Minimum word count for inclusion in the model.
        model_type (str): Type of Word2Vec model - 'cbow' or 'skipgram'.

    Returns:
        tuple: A tuple containing the trained models and the DataFrame with topic assignments.
    """
    print("\n" + "-" * 80)
    if model_type.lower() == 'skipgram':
        print("RUNNING WORD2VEC SKIP-GRAM TOPIC MODELING")
        sg_param = 1  # 1 for Skip-gram
        topic_col_name = 'word2vec_skipgram_topic'
        topic_name_col = 'word2vec_skipgram_topic_name'
    else:
        print("RUNNING WORD2VEC CBOW TOPIC MODELING")
        sg_param = 0  # 0 for CBOW
        topic_col_name = 'word2vec_topic'
        topic_name_col = 'word2vec_topic_name'
    print("-" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data Preparation ---
    df_topics = df.dropna(subset=['description_clean']).copy()
    df_topics = df_topics[df_topics['description_clean'].str.strip() != ''].copy()
    documents = df_topics['description_clean'].tolist()
    print(f"  Filtered to {len(documents):,} articles with valid descriptions for topic modeling.")

    # --- Preprocess texts for Word2Vec ---
    processed_texts = preprocess_for_word2vec(documents)
    print(f"  Preprocessed {len(processed_texts)} documents for Word2Vec.")

    if len(processed_texts) == 0:
        print("  ! Warning: No valid documents found for topic modeling.")
        return None, df

    # --- Train Word2Vec Model ---
    word2vec_model = train_word2vec_model(
        processed_texts,
        vector_size=vector_size,
        min_count=min_count,
        sg=sg_param
    )

    # --- Compute Document Vectors ---
    doc_vectors = get_document_vectors(df_topics, word2vec_model)

    # --- Cluster Documents ---
    cluster_labels, kmeans_model = cluster_documents(doc_vectors, n_clusters)

    # --- Extract Top Words Per Cluster ---
    cluster_to_words = extract_words_per_cluster(df_topics, cluster_labels, word2vec_model)
    
    # --- Save Models ---
    word2vec_model_path = output_dir / "word2vec_model.model"
    kmeans_model_path = output_dir / "kmeans_model.pkl"
    
    word2vec_model.save(str(word2vec_model_path))
    with open(kmeans_model_path, 'wb') as f:
        pickle.dump(kmeans_model, f)
    
    print(f"  ✓ Word2Vec model saved to {word2vec_model_path}")
    print(f"  ✓ K-means model saved to {kmeans_model_path}")

    # --- Process Results ---
    df_topics[topic_col_name] = cluster_labels
    df_topics[topic_name_col] = df_topics[topic_col_name].apply(
        lambda t: ", ".join(cluster_to_words.get(t, ["unknown"])[:5])
    )
    print(f"  ✓ Assigned Word2Vec {model_type} topics back to DataFrame.")
    
    # --- Print topic summaries ---
    print("\nTopic summaries:")
    for cluster_id in sorted(cluster_to_words.keys()):
        top_words = cluster_to_words[cluster_id][:10]
        print(f"  Topic {cluster_id}: {' | '.join(top_words)}")

    # --- Generate and Save Visualizations ---
    print("  Generating and saving Word2Vec topic visualizations...")
    
    # Save cluster information
    cluster_info_path = output_dir / "word2vec_cluster_info.pkl"
    with open(cluster_info_path, 'wb') as f:
        pickle.dump({
            'cluster_labels': cluster_labels,
            'cluster_to_words': cluster_to_words,
            'n_clusters': n_clusters
        }, f)
    print(f"  ✓ Cluster information saved to {cluster_info_path}")

    return {
        'word2vec_model': word2vec_model,
        'kmeans_model': kmeans_model,
        'cluster_labels': cluster_labels,
        'cluster_to_words': cluster_to_words
    }, df_topics


if __name__ == "__main__":
    # For testing purposes
    print("Word2Vec CBOW topic modeling module loaded successfully.")