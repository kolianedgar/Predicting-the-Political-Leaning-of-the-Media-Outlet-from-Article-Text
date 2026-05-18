"""
Feature extraction utilities for political article classification.

This module provides:
- TF-IDF lexical features
- NMF topic features
- Linguistic/POS features
- Sentiment features
- Readability metrics
- Sentence embeddings
- Chunked document embeddings

The feature builders are designed for:
- article-level binary classification
- grouped/source-aware evaluation
- external full-text datasets

All functions assume English-language article text.
"""

import numpy as np
import spacy
import textstat

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF

from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer


# =========================
# CONFIGURATION
# =========================

SEED = 42

TFIDF_MAX_FEATURES = 30000
TFIDF_NGRAM_RANGE = (1, 2)

NMF_COMPONENTS = 20
NMF_MAX_ITER = 500

CHUNK_SIZE = 200

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

MODALS = {
    "must",
    "should",
    "may",
    "might",
    "could",
    "would",
    "can"
}


# =========================
# GLOBAL NLP OBJECTS
# =========================

nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser"]
)

sia = SentimentIntensityAnalyzer()

model_emb = SentenceTransformer(EMBEDDING_MODEL_NAME)


# =========================
# INTERNAL HELPERS
# =========================

def safe_text(text):
    """
    Ensure text is a valid non-null string.
    """

    if text is None:
        return ""

    return str(text).strip()


def validate_texts(texts):
    """
    Validate text input collection.
    """

    if texts is None:
        raise ValueError("Input texts cannot be None.")

    if len(texts) == 0:
        raise ValueError("Input text collection is empty.")


# =========================
# LINGUISTIC FEATURES
# =========================

def extract_pos_features_batch(texts):
    """
    Extract normalized POS and modal verb frequencies.

    Features:
    - noun ratio
    - verb ratio
    - adjective ratio
    - adverb ratio
    - pronoun ratio
    - modal ratio

    Returns
    -------
    numpy.ndarray
        Shape: (n_documents, 6)
    """

    validate_texts(texts)

    features = []

    for text in texts:

        text = safe_text(text)

        doc = nlp(text)

        total_tokens = 0

        noun = 0
        verb = 0
        adj = 0
        adv = 0
        pron = 0
        modal = 0

        for token in doc:

            if token.is_punct or token.is_space:
                continue

            total_tokens += 1

            if token.pos_ == "NOUN":
                noun += 1

            elif token.pos_ == "VERB":
                verb += 1

            elif token.pos_ == "ADJ":
                adj += 1

            elif token.pos_ == "ADV":
                adv += 1

            elif token.pos_ == "PRON":
                pron += 1

            if token.lemma_.lower() in MODALS:
                modal += 1

        if total_tokens == 0:

            features.append([0.0] * 6)

        else:

            features.append([
                noun / total_tokens,
                verb / total_tokens,
                adj / total_tokens,
                adv / total_tokens,
                pron / total_tokens,
                modal / total_tokens
            ])

    return np.array(features)


def extract_sentiment_batch(texts):
    """
    Extract VADER sentiment features.

    Features:
    - negative
    - neutral
    - positive
    - compound

    Returns
    -------
    numpy.ndarray
        Shape: (n_documents, 4)
    """

    validate_texts(texts)

    sentiment_features = []

    for text in texts:

        text = safe_text(text)

        scores = sia.polarity_scores(text[:1000])

        sentiment_features.append([
            scores["neg"],
            scores["neu"],
            scores["pos"],
            scores["compound"]
        ])

    return np.array(sentiment_features)


def extract_readability_batch(texts):
    """
    Extract readability metrics.

    Features:
    - Flesch Reading Ease
    - Flesch-Kincaid Grade
    - Gunning Fog
    - SMOG Index
    - Automated Readability Index
    - Words per Sentence

    Returns
    -------
    numpy.ndarray
        Shape: (n_documents, 6)
    """

    validate_texts(texts)

    readability_features = []

    for text in texts:

        text = safe_text(text)

        readability_features.append([
            textstat.flesch_reading_ease(text),
            textstat.flesch_kincaid_grade(text),
            textstat.gunning_fog(text),
            textstat.smog_index(text),
            textstat.automated_readability_index(text),
            textstat.words_per_sentence(text)
        ])

    return np.array(readability_features)


# =========================
# TF-IDF FEATURES
# =========================

def build_tfidf(train_text, test_text):
    """
    Build TF-IDF unigram/bigram features.

    Returns
    -------
    tuple
        Sparse train/test matrices.
    """

    validate_texts(train_text)
    validate_texts(test_text)

    n_docs = len(train_text)

    min_df = 3 if n_docs >= 10 else 1
    max_df = 0.85 if n_docs >= 10 else 1.0

    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=min_df,
        max_df=max_df,
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    return X_train, X_test


# =========================
# TF-IDF + NMF
# =========================

def build_tfidf_nmf(train_text, test_text):
    """
    Combine TF-IDF lexical features with
    NMF topic representations.

    Returns
    -------
    tuple
        Sparse hybrid feature matrices.
    """

    X_train_tf, X_test_tf = build_tfidf(
        train_text,
        test_text
    )

    nmf = NMF(
        n_components=NMF_COMPONENTS,
        random_state=SEED,
        max_iter=NMF_MAX_ITER
    )

    W_train = nmf.fit_transform(X_train_tf)
    W_test = nmf.transform(X_test_tf)

    return (
        hstack([X_train_tf, W_train]),
        hstack([X_test_tf, W_test])
    )


# =========================
# FULL FEATURE SET
# =========================

def build_full(train_text, test_text):
    """
    Combine:
    - TF-IDF lexical features
    - POS features
    - sentiment features
    - readability features

    Returns
    -------
    tuple
        Sparse hybrid train/test matrices.
    """

    X_train_tf, X_test_tf = build_tfidf(
        train_text,
        test_text
    )

    X_train_extra = np.hstack([
        extract_pos_features_batch(train_text),
        extract_sentiment_batch(train_text),
        extract_readability_batch(train_text)
    ])

    X_test_extra = np.hstack([
        extract_pos_features_batch(test_text),
        extract_sentiment_batch(test_text),
        extract_readability_batch(test_text)
    ])

    scaler = StandardScaler()

    X_train_extra = scaler.fit_transform(X_train_extra)
    X_test_extra = scaler.transform(X_test_extra)

    return (
        hstack([X_train_tf, csr_matrix(X_train_extra)]),
        hstack([X_test_tf, csr_matrix(X_test_extra)])
    )


# =========================
# SENTENCE EMBEDDINGS
# =========================

def build_embeddings(train_text, test_text):
    """
    Generate dense sentence embeddings
    for full documents.

    Returns
    -------
    tuple
        Dense embedding matrices.
    """

    validate_texts(train_text)
    validate_texts(test_text)

    return (
        model_emb.encode(
            list(train_text),
            show_progress_bar=False
        ),

        model_emb.encode(
            list(test_text),
            show_progress_bar=False
        )
    )


# =========================
# CHUNKED EMBEDDINGS
# =========================

def chunk_text(text, size=CHUNK_SIZE):
    """
    Split long documents into word chunks.
    """

    text = safe_text(text)

    words = text.split()

    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]


def embed_document(text):
    """
    Generate averaged chunk embeddings
    for a single document.
    """

    chunks = chunk_text(text)

    if len(chunks) == 0:

        return np.zeros(
            model_emb.get_sentence_embedding_dimension()
        )

    embeddings = model_emb.encode(
        chunks,
        show_progress_bar=False
    )

    return np.mean(embeddings, axis=0)


def build_chunked(train_text, test_text):
    """
    Generate chunk-averaged embeddings
    for train/test datasets.

    Returns
    -------
    tuple
        Dense embedding matrices.
    """

    validate_texts(train_text)
    validate_texts(test_text)

    return (
        np.array([
            embed_document(text)
            for text in train_text
        ]),

        np.array([
            embed_document(text)
            for text in test_text
        ])
    )


# =========================
# FEATURE REGISTRY
# =========================

FEATURE_BUILDERS = {
    "tfidf": build_tfidf,
    "tfidf_nmf": build_tfidf_nmf,
    "full": build_full,
    "embeddings": build_embeddings,
    "chunked": build_chunked
}