# 구현하세요!
from datasets import load_dataset
from typing import List

def load_corpus() -> List[str]:
    """
    Load the corpus from the Poem Sentiment dataset.
    Returns:
        List[str]: List of verses from the dataset.
    """
    # Load the Poem Sentiment dataset
    ds = load_dataset("google-research-datasets/poem_sentiment")

    # Extract the training and validation texts
    train_texts = list(ds["train"]["verse_text"])
    val_texts   = list(ds["validation"]["verse_text"])

    corpus = [t for t in (train_texts + val_texts) if t.strip()]
    return corpus
