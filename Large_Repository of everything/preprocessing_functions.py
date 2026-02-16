# preprocess.py
# Complete preprocessing pipeline for RNN sentiment analysis


import pandas as pd
import re
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Contractions Dictionary
# --------------------------------------------------

CONTRACTIONS = {
    "don't": "do not",
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am"
}


# --------------------------------------------------
# Negation Words
# --------------------------------------------------

NEGATION_WORDS = {
    "not", "no", "never", "none", "cannot"
}


# --------------------------------------------------
# Expand Contractions
# --------------------------------------------------

def expand_contractions(text):

    for key, val in CONTRACTIONS.items():
        text = re.sub(key, val, text)

    return text


# --------------------------------------------------
# Clean Text
# --------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = expand_contractions(text)

    # Remove symbols and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# Handle Negation
# --------------------------------------------------

def handle_negation(text, window=3):

    words = text.split()

    result = []

    negate = 0

    for word in words:

        if word in NEGATION_WORDS:
            negate = window
            result.append(word)
            continue

        if negate > 0:
            result.append("NEG_" + word)
            negate -= 1

        else:
            result.append(word)

    return " ".join(result)


# --------------------------------------------------
# Preprocess Single Sentence (For Testing)
# --------------------------------------------------

def preprocess_text(text, tokenizer, max_len=30):

    text = clean_text(text)

    text = handle_negation(text)

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    return padded


# --------------------------------------------------
# Main Preprocessing Pipeline
# --------------------------------------------------

def preprocess_dataset(
    csv_path,
    vocab_size=10000,
    max_len=30,
    test_size=0.2,
    random_state=42
):
    """
    Loads CSV dataset and preprocesses it for RNN.

    Returns:
    X_train, X_test, y_train, y_test,
    tokenizer, label_encoder
    """

    # ----------------------------------------------
    # Load Data (with encoding fix)
    # ----------------------------------------------

    df = pd.read_csv(
    csv_path,
    sep=",",
    engine="python",
    encoding="latin1"
    )


    # ----------------------------------------------
    # Check Required Columns
    # ----------------------------------------------

    if "phrase" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            "CSV must contain 'phrase' and 'sentiment' columns"
        )

    # ----------------------------------------------
    # Clean Text
    # ----------------------------------------------

    df["clean_text"] = df["phrase"].apply(clean_text)

    # ----------------------------------------------
    # Handle Negation
    # ----------------------------------------------

    df["processed_text"] = df["clean_text"].apply(handle_negation)

    # ----------------------------------------------
    # Tokenization
    # ----------------------------------------------

    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["processed_text"])

    sequences = tokenizer.texts_to_sequences(
        df["processed_text"]
    )

    # ----------------------------------------------
    # Padding
    # ----------------------------------------------

    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    # ----------------------------------------------
    # Encode Labels
    # ----------------------------------------------

    encoder = LabelEncoder()

    labels = encoder.fit_transform(
        df["sentiment"]
    )

    # ----------------------------------------------
    # Check if stratification is possible
    # ----------------------------------------------
    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    
    if min_count < 2:
        print("⚠️ Warning: Some classes have <2 samples. Stratify disabled.")
        stratify_labels = None
    else:
        stratify_labels = labels
    
    
    # ----------------------------------------------
    # Train / Test Split
    # ----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        padded,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )


    # ----------------------------------------------
    # Return Everything
    # ----------------------------------------------

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        encoder
    )


# --------------------------------------------------
# Run Test (Optional: For Debugging)
# --------------------------------------------------

if __name__ == "__main__":

    print("Running preprocessing test...")

    X_train, X_test, y_train, y_test, tokenizer, encoder = preprocess_dataset(
        "ds.csv"
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Classes:", encoder.classes_)

    sample = "I do not like this movie at all"

    x = preprocess_text(sample, tokenizer)

    print("Sample input:", sample)
    print("Processed:", x)