import spacy
import pandas as pd
from tqdm import tqdm

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def perform_ner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Named Entity Recognition (NER) on the 'description' column of the DataFrame.

    Args:
        df: The input DataFrame with a 'description' column.

    Returns:
        The DataFrame with an added 'entities' column.
    """
    tqdm.pandas(desc="Performing NER")
    df['entities'] = df['description'].progress_apply(lambda text: [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in nlp(text).ents])
    return df
