"""
preprocessing.py

Utilities for loading, standardizing, validating, and preparing datasets
for political article classification experiments.

This module supports two dataset modes:

1. Metadata-only placeholder datasets
   - Public GitHub-safe datasets
   - Contain article metadata only
   - Used for reproducibility, inspection, and documentation
   - Do NOT contain article bodies

2. Full-text datasets
   - External or privately held corpora
   - Contain complete article text
   - Used for feature extraction and classification experiments

The preprocessing pipeline is designed to:
- normalize heterogeneous dataset schemas
- support grouped/source-aware evaluation
- preserve source metadata
- maintain compatibility with external datasets
  such as AG News or custom news corpora

Repository note:
Public repository datasets may omit article body text
for copyright and redistribution reasons.
"""


# ======================
# IMPORTS
# ======================

import json
import os

import pandas as pd


# ======================
# GLOBAL CONFIGURATION
# ======================

SEED = 42

DEFAULT_DATA_PATH = "data/articles"

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_SOURCE_COLUMN = "source"


# ======================
# POLITICAL LABEL GROUPS
# ======================

LEFT_SOURCES = {
    "AlterNet",
    "CounterPunch",
    "DailyBeast",
    "MotherJones",
    "Salon"
}

RIGHT_SOURCES = {
    "American_Conservative",
    "American_Spectator",
    "DailyCaller",
    "FoxNews",
    "The_Free_Beacon"
}

KNOWN_SOURCES = LEFT_SOURCES.union(RIGHT_SOURCES)


# ======================
# DATASET FILE REGISTRY
# ======================

DATASET_FILES = {
    "AlterNet": "left/alter-net/alternet_200_articles.json",
    "CounterPunch": "left/counter-punch/counterpunch_200_articles.json",
    "DailyBeast": "left/daily-beast/dailybeast_200_articles.json",
    "MotherJones": "left/mother-jones/mother_jones_articles.json",
    "Salon": "left/salon/salon_articles.json",
    "American_Conservative": "right/american-conservative/the_american_conservative.json",
    "American_Spectator": "right/american-spectator/spectator_articles.json",
    "DailyCaller": "right/daily-caller/dailycaller_200_articles.json",
    "FoxNews": "right/fox-news/foxnews_200_articles.json",
    "The_Free_Beacon": "right/the-free-beacon/freebeacon_articles.json"
}


# ======================
# STANDARDIZED SCHEMA
# ======================

STANDARD_COLUMNS = {
    "article_id",
    "url",
    "headline",
    "publication_date",
    "outlet_name",
    "word_count",
    "label",
    "crawl_timestamp",
    "text"
}


# ======================
# COLUMN NORMALIZATION MAP
# ======================

COLUMN_ALIASES = {

    # ------------------
    # ARTICLE ID
    # ------------------
    "article_id": [
        "article_id",
        "articleid",
        "id",
        "doc_id",
        "document_id",
        "news_id",
        "uuid",
        "uid"
    ],

    # ------------------
    # URL
    # ------------------
    "url": [
        "url",
        "article_url",
        "link",
        "source_url",
        "web_url",
        "page_url"
    ],

    # ------------------
    # HEADLINE / TITLE
    # ------------------
    "headline": [
        "headline",
        "title",
        "header",
        "article_title",
        "news_title"
    ],

    # ------------------
    # PUBLICATION DATE
    # ------------------
    "publication_date": [
        "publication_date",
        "publish_date",
        "published_at",
        "published_date",
        "date",
        "created_at",
        "article_date",
        "timestamp"
    ],

    # ------------------
    # OUTLET / SOURCE
    # ------------------
    "outlet_name": [
        "outlet_name",
        "outlet",
        "source",
        "source_name",
        "publisher",
        "publication",
        "news_source"
    ],

    # ------------------
    # WORD COUNT
    # ------------------
    "word_count": [
        "word_count",
        "body_word_count",
        "token_count",
        "article_word_count",
        "wc"
    ],

    # ------------------
    # LABEL
    # ------------------
    "label": [
        "label",
        "class",
        "target",
        "category",
        "y"
    ],

    # ------------------
    # CRAWL TIMESTAMP
    # ------------------
    "crawl_timestamp": [
        "crawl_timestamp",
        "crawl_date",
        "scrape_timestamp",
        "scraped_at",
        "retrieved_at",
        "downloaded_at",
        "collection_timestamp"
    ],

    # ------------------
    # FULL ARTICLE TEXT
    # ------------------
    "text": [
        "text",
        "body",
        "body_text",
        "article_text",
        "content",
        "main_text",
        "main_body",
        "full_text",
        "article_body",
        "news_text"
    ]
}


# ======================
# REQUIRED COLUMNS
# ======================

METADATA_REQUIRED_COLUMNS = {
    "headline",
    "outlet_name"
}

FULLTEXT_REQUIRED_COLUMNS = {
    "text",
    "label"
}


# ======================
# DATASET MODES
# ======================

DATASET_MODE_METADATA = "metadata_only"

DATASET_MODE_FULLTEXT = "full_text"


# ======================
# TEXT CLEANING DEFAULTS
# ======================

DEFAULT_LOWERCASE = False
DEFAULT_STRIP_WHITESPACE = True
DEFAULT_DROP_EMPTY_TEXT = True

# ======================
# INTERNAL NORMALIZATION HELPERS
# ======================

def normalize_string_series(series):
    """
    Normalize a pandas Series containing string-like values.

    Processing steps:
    - replace missing values with empty strings
    - convert to string dtype
    - strip surrounding whitespace

    Parameters
    ----------
    series : pandas.Series
        Input series.

    Returns
    -------
    pandas.Series
        Normalized string series.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

# ======================
# FULL-TEXT PREPROCESSING HELPERS
# ======================

def normalize_text_column(
    df,
    text_column="text",
    lowercase=DEFAULT_LOWERCASE,
    strip_whitespace=DEFAULT_STRIP_WHITESPACE
):
    """
    Normalize article text content.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    text_column : str, default="text"
        Text column name.

    lowercase : bool, default=False
        Whether to lowercase text.

    strip_whitespace : bool, default=True
        Whether to strip surrounding whitespace.

    Returns
    -------
    pandas.DataFrame
        Dataframe with normalized text column.
    """

    df = df.copy()

    if text_column not in df.columns:
        raise ValueError(
            f"Missing text column: '{text_column}'"
        )

    text_series = normalize_string_series(
        df[text_column]
    )

    if not strip_whitespace:
        text_series = (
            df[text_column]
            .fillna("")
            .astype(str)
        )

    if lowercase:
        text_series = text_series.str.lower()

    df[text_column] = text_series

    return df


def ensure_source_column(
    df,
    source_col="outlet_name",
    default_source="external"
):
    """
    Ensure a standardized source column exists.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    source_col : str, default="outlet_name"
        Standardized source column name.

    default_source : str, default="external"
        Default source value if column is missing.

    Returns
    -------
    pandas.DataFrame
        Dataframe with guaranteed source column.
    """

    df = df.copy()

    if source_col not in df.columns:
        df[source_col] = default_source

    df = normalize_source_column(
        df,
        source_col=source_col
    )

    return df


def remove_empty_text_rows(
    df,
    text_column="text",
    verbose=True
):
    """
    Remove rows containing empty article text.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    text_column : str, default="text"
        Text column name.

    verbose : bool, default=True
        Whether to print removal statistics.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe.

    Raises
    ------
    ValueError
        If all rows are removed.
    """

    df = df.copy()

    initial_size = len(df)

    mask = get_nonempty_text_mask(
        df,
        text_column=text_column
    )

    df = df[mask]

    removed_rows = initial_size - len(df)

    if len(df) == 0:
        raise ValueError(
            "All rows were removed because "
            "article text is empty."
        )

    if verbose and removed_rows > 0:
        print(
            f"Removed {removed_rows} "
            f"empty-text rows."
        )

    return df


def keep_standard_columns(
    df,
    prioritize=None
):
    """
    Retain standardized project columns while preserving order.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    prioritize : list or None, optional
        Columns to place first.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe.
    """

    prioritize = prioritize or []

    prioritized_columns = [
        col for col in prioritize
        if col in df.columns
    ]

    remaining_standard_columns = [
        col for col in STANDARD_COLUMNS
        if (
            col in df.columns
            and col not in prioritized_columns
        )
    ]

    final_columns = (
        prioritized_columns
        + remaining_standard_columns
    )

    return df[final_columns]

def normalize_numeric_series(series):
    """
    Normalize a numeric pandas Series.

    Non-numeric values are coerced to NaN.

    Parameters
    ----------
    series : pandas.Series
        Input series.

    Returns
    -------
    pandas.Series
        Numeric series.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ======================
# INTERNAL SCHEMA HELPERS
# ======================

def detect_standard_columns(df):
    """
    Detect standardized schema columns using COLUMN_ALIASES.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    dict
        Mapping of standardized column names
        to detected dataframe columns.
    """

    detected_columns = {}

    for standard_col, aliases in COLUMN_ALIASES.items():

        matched_col = None

        for alias in aliases:

            if alias in df.columns:
                matched_col = alias
                break

        detected_columns[standard_col] = matched_col

    return detected_columns


def get_existing_standard_columns(df):
    """
    Return standardized columns that currently exist
    in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    list
        Existing standardized columns.
    """

    return [
        col for col in STANDARD_COLUMNS
        if col in df.columns
    ]


# ======================
# INTERNAL DATASET HELPERS
# ======================

def get_nonempty_text_mask(df, text_column="text"):
    """
    Generate a boolean mask for rows containing
    non-empty text.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    text_column : str, default="text"
        Name of text column.

    Returns
    -------
    pandas.Series
        Boolean mask.
    """

    if text_column not in df.columns:
        return pd.Series(False, index=df.index)

    normalized_text = normalize_string_series(
        df[text_column]
    )

    return normalized_text != ""


def count_nonempty_text_rows(df, text_column="text"):
    """
    Count rows containing non-empty text.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    text_column : str, default="text"
        Name of text column.

    Returns
    -------
    int
        Number of rows with usable text.
    """

    return int(
        get_nonempty_text_mask(
            df,
            text_column=text_column
        ).sum()
    )


def normalize_source_column(df, source_col="outlet_name"):
    """
    Normalize source/outlet column values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    source_col : str, default="outlet_name"
        Source column name.

    Returns
    -------
    pandas.DataFrame
        Dataframe with normalized source column.
    """

    df = df.copy()

    if source_col in df.columns:

        df[source_col] = normalize_string_series(
            df[source_col]
        )

    return df

# ======================
# BASIC JSON LOADING
# ======================

def load_json_file(filepath, normalize=True):
    """
    Load a JSON dataset file into a pandas DataFrame.

    Supports both:
    - metadata-only placeholder datasets
    - full-text article datasets

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    normalize : bool, default=True
        Whether to apply pandas.json_normalize()
        to nested JSON structures.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If the JSON file is empty or invalid.

    Notes
    -----
    Public repository datasets may contain metadata only
    and therefore may not include article body text.
    """

    # ------------------
    # FILE VALIDATION
    # ------------------

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}"
        )

    if not filepath.lower().endswith(".json"):
        raise ValueError(
            f"Expected a JSON file, got: {filepath}"
        )

    # ------------------
    # LOAD JSON CONTENT
    # ------------------

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON format in file: {filepath}"
        ) from e

    # ------------------
    # EMPTY FILE CHECK
    # ------------------

    if data is None:
        raise ValueError(
            f"JSON file is empty: {filepath}"
        )

    if isinstance(data, list) and len(data) == 0:
        raise ValueError(
            f"JSON dataset contains no records: {filepath}"
        )

    # ------------------
    # DATAFRAME CREATION
    # ------------------

    if normalize:
        df = pd.DataFrame(pd.json_normalize(data))
    else:
        df = pd.DataFrame(data)

    # ------------------
    # FINAL VALIDATION
    # ------------------

    if df.empty:
        raise ValueError(
            f"Loaded dataframe is empty: {filepath}"
        )

    return df

# ======================
# STANDARDIZATION
# ======================

def standardize_columns(df):
    """
    Standardize heterogeneous dataset schemas into a unified format.

    This function maps multiple possible column name variations
    to a consistent standardized schema defined in COLUMN_ALIASES.

    The function supports both:
    - metadata-only datasets
    - full-text datasets

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with arbitrary column naming conventions.

    Returns
    -------
    pandas.DataFrame
        Dataframe with standardized column names.

    Notes
    -----
    Existing standardized columns are preserved.
    Unknown columns are retained unchanged.
    """

    df = df.copy()

    # ------------------
    # APPLY COLUMN ALIASES
    # ------------------

    for standard_col, aliases in COLUMN_ALIASES.items():

        # Skip if standardized column already exists
        if standard_col in df.columns:
            continue

        for alias in aliases:

            if alias in df.columns:
                df[standard_col] = df[alias]
                break

    # ------------------
    # OPTIONAL COLUMN ORDERING
    # ------------------

    standardized_cols = [
        col for col in STANDARD_COLUMNS
        if col in df.columns
    ]

    remaining_cols = [
        col for col in df.columns
        if col not in standardized_cols
    ]

    df = df[standardized_cols + remaining_cols]

    return df

def clean_metadata_dataframe(df):
    """
    Clean and normalize a metadata-only dataset.

    This function is intended for public repository datasets that
    contain article metadata but do not necessarily include article
    body text.

    Processing steps:
    - standardize column names
    - retain recognized metadata columns
    - normalize missing values
    - strip whitespace
    - remove fully empty rows
    - reset dataframe index

    Parameters
    ----------
    df : pandas.DataFrame
        Input metadata dataframe.

    Returns
    -------
    pandas.DataFrame
        Cleaned metadata dataframe.
    """

    if df is None or len(df) == 0:
        raise ValueError("Input dataframe is empty.")

    df = df.copy()

    # ======================
    # STANDARDIZE COLUMNS
    # ======================

    df = standardize_columns(df)

    # ======================
    # RETAIN ONLY KNOWN COLUMNS
    # ======================

    keep_columns = [
        col for col in STANDARD_COLUMNS
        if col in df.columns
    ]

    df = df[keep_columns]

    # ======================
    # ENSURE REQUIRED COLUMNS
    # ======================

    missing_required = [
        col for col in METADATA_REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required metadata columns: {missing_required}"
        )

    # ======================
    # NORMALIZE STRING FIELDS
    # ======================

    string_columns = [
        "article_id",
        "url",
        "headline",
        "publication_date",
        "outlet_name",
        "crawl_timestamp",
        "text"
    ]

    for col in string_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # ======================
    # NORMALIZE NUMERIC FIELDS
    # ======================

    if "word_count" in df.columns:

        df["word_count"] = pd.to_numeric(
            df["word_count"],
            errors="coerce"
        )

    if "label" in df.columns:

        df["label"] = pd.to_numeric(
            df["label"],
            errors="coerce"
        )

    # ======================
    # REMOVE FULLY EMPTY ROWS
    # ======================

    essential_columns = [
        col for col in [
            "headline",
            "url",
            "article_id"
        ]
        if col in df.columns
    ]

    if essential_columns:

        df = df[
            df[essential_columns]
            .astype(str)
            .apply(
                lambda row: any(
                    value.strip() != ""
                    for value in row
                ),
                axis=1
            )
        ]

    # ======================
    # RESET INDEX
    # ======================

    df = df.reset_index(drop=True)

    return df

def dataset_has_text(df, text_column="text", min_nonempty_ratio=0.1):
    """
    Determine whether a dataset contains usable article body text.

    This function distinguishes between:
    - metadata-only placeholder datasets
    - full-text trainable datasets

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    text_column : str, default="text"
        Standardized text column name.

    min_nonempty_ratio : float, default=0.1
        Minimum proportion of non-empty text rows required
        to classify the dataset as full-text.

    Returns
    -------
    bool
        True if usable article text is present.
        False otherwise.
    """

    # ----------------------
    # Missing text column
    # ----------------------
    if text_column not in df.columns:
        return False

    # ----------------------
    # Convert safely
    # ----------------------
    text_series = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ----------------------
    # Empty dataframe
    # ----------------------
    if len(text_series) == 0:
        return False

    # ----------------------
    # Count usable rows
    # ----------------------
    nonempty_count = (text_series != "").sum()

    nonempty_ratio = nonempty_count / len(text_series)

    # ----------------------
    # Metadata-only detection
    # ----------------------
    return nonempty_ratio >= min_nonempty_ratio

def inspect_dataset_schema(df):
    """
    Inspect dataset structure and determine whether the dataset
    is metadata-only or suitable for full-text NLP experiments.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    dict
        Dictionary containing:
        - dataset mode
        - row/column counts
        - standardized columns
        - missing required fields
        - text availability
        - metadata summary
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # ---------------------
    # BASIC STRUCTURE
    # ---------------------

    n_rows, n_cols = df.shape

    original_columns = list(df.columns)

    detected_standard_columns = {}

    # ---------------------
    # DETECT COLUMN MAPPINGS
    # ---------------------

    for standard_col, aliases in COLUMN_ALIASES.items():

        matched_col = None

        for alias in aliases:
            if alias in df.columns:
                matched_col = alias
                break

        detected_standard_columns[standard_col] = matched_col

    # ---------------------
    # TEXT AVAILABILITY
    # ---------------------

    text_col = detected_standard_columns.get("text")

    has_text_column = text_col is not None

    has_usable_text = False

    non_empty_text_rows = 0

    if has_text_column:

        non_empty_text_rows = (
            df[text_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        has_usable_text = non_empty_text_rows > 0

    # ---------------------
    # DATASET MODE
    # ---------------------

    dataset_mode = (
        DATASET_MODE_FULLTEXT
        if has_usable_text
        else DATASET_MODE_METADATA
    )

    # ---------------------
    # REQUIRED COLUMN CHECKS
    # ---------------------

    missing_metadata_columns = []

    for required_col in METADATA_REQUIRED_COLUMNS:
        if detected_standard_columns.get(required_col) is None:
            missing_metadata_columns.append(required_col)

    missing_fulltext_columns = []

    for required_col in FULLTEXT_REQUIRED_COLUMNS:
        if detected_standard_columns.get(required_col) is None:
            missing_fulltext_columns.append(required_col)

    # ---------------------
    # SOURCE SUMMARY
    # ---------------------

    outlet_col = detected_standard_columns.get("outlet_name")

    unique_sources = []

    n_sources = 0

    if outlet_col is not None:

        unique_sources = sorted(
            df[outlet_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        n_sources = len(unique_sources)

    # ---------------------
    # LABEL SUMMARY
    # ---------------------

    label_col = detected_standard_columns.get("label")

    label_distribution = None

    if label_col is not None:

        label_distribution = (
            df[label_col]
            .value_counts(dropna=False)
            .to_dict()
        )

    # ---------------------
    # WORD COUNT SUMMARY
    # ---------------------

    word_count_col = detected_standard_columns.get("word_count")

    word_count_stats = None

    if word_count_col is not None:

        numeric_wc = pd.to_numeric(
            df[word_count_col],
            errors="coerce"
        )

        word_count_stats = {
            "min": float(numeric_wc.min()) if not numeric_wc.isna().all() else None,
            "max": float(numeric_wc.max()) if not numeric_wc.isna().all() else None,
            "mean": float(numeric_wc.mean()) if not numeric_wc.isna().all() else None
        }

    # ---------------------
    # FINAL REPORT
    # ---------------------

    report = {
        "dataset_mode": dataset_mode,
        "n_rows": n_rows,
        "n_columns": n_cols,
        "original_columns": original_columns,
        "detected_standard_columns": detected_standard_columns,
        "has_text_column": has_text_column,
        "has_usable_text": has_usable_text,
        "non_empty_text_rows": int(non_empty_text_rows),
        "missing_metadata_columns": missing_metadata_columns,
        "missing_fulltext_columns": missing_fulltext_columns,
        "n_sources": n_sources,
        "sources": unique_sources,
        "label_distribution": label_distribution,
        "word_count_stats": word_count_stats
    }

    return report

# ======================
# FULL-TEXT DATASET VALIDATION
# ======================

def validate_text_dataset(
    df,
    text_column="text",
    label_column="label",
    min_samples=10,
    require_binary_labels=True
):
    """
    Validate whether a dataframe is suitable for
    full-text classification experiments.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    text_column : str, default="text"
        Name of the article text column.

    label_column : str, default="label"
        Name of the target label column.

    min_samples : int, default=10
        Minimum required number of usable samples.

    require_binary_labels : bool, default=True
        Whether to enforce binary classification labels.

    Returns
    -------
    pandas.DataFrame
        Validated and cleaned dataframe.

    Raises
    ------
    ValueError
        If the dataset is unsuitable for training.
    """

    # ======================
    # BASIC VALIDATION
    # ======================

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    # ======================
    # REQUIRED COLUMNS
    # ======================

    missing_columns = [
        col for col in [text_column, label_column]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ======================
    # TEXT AVAILABILITY
    # ======================

    if not dataset_has_text(
        df,
        text_column=text_column
    ):
        raise ValueError(
            "Dataset contains no usable article text."
        )

    # ======================
    # NORMALIZE TEXT
    # ======================

    df = df.copy()

    df[text_column] = normalize_string_series(
        df[text_column]
    )

    # ======================
    # REMOVE EMPTY TEXT ROWS
    # ======================

    initial_size = len(df)

    nonempty_mask = get_nonempty_text_mask(
        df,
        text_column=text_column
    )

    df = df[nonempty_mask]

    removed_rows = initial_size - len(df)

    # ======================
    # SAMPLE COUNT CHECK
    # ======================

    if len(df) < min_samples:
        raise ValueError(
            f"Dataset contains too few usable samples "
            f"({len(df)} found, minimum={min_samples})."
        )

    # ======================
    # LABEL VALIDATION
    # ======================

    if df[label_column].isna().any():
        raise ValueError(
            "Label column contains missing values."
        )

    unique_labels = sorted(
        df[label_column].unique()
    )

    if len(unique_labels) < 2:
        raise ValueError(
            "Dataset must contain at least two classes."
        )

    if (
        require_binary_labels
        and len(unique_labels) != 2
    ):
        raise ValueError(
            f"Binary classification expected, "
            f"but found labels: {unique_labels}"
        )

    # ======================
    # OPTIONAL SOURCE CHECK
    # ======================

    if "outlet_name" in df.columns:

        normalized_sources = normalize_string_series(
            df["outlet_name"]
        )

        missing_sources = (
            normalized_sources == ""
        ).sum()

        if missing_sources > 0:

            print(
                f"Warning: {missing_sources} rows "
                f"contain missing outlet names."
            )

    # ======================
    # VALIDATION SUMMARY
    # ======================

    print("Dataset validation successful.")
    print(f"Samples retained: {len(df)}")

    if removed_rows > 0:
        print(
            f"Removed empty-text rows: "
            f"{removed_rows}"
        )

    print(f"Unique labels: {unique_labels}")

    return df

# ======================
# LABEL ENCODING
# ======================

def encode_labels(
    df,
    source_col="outlet_name",
    label_col="label",
    inplace=False
):
    """
    Encode binary political leaning labels
    based on outlet/source names.

    Label mapping:
    - Left-leaning outlets  -> 0
    - Right-leaning outlets -> 1

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing source metadata.

    source_col : str, default="outlet_name"
        Column containing outlet/source names.

    label_col : str, default="label"
        Name of output label column.

    inplace : bool, default=False
        Whether to modify dataframe in place.

    Returns
    -------
    pandas.DataFrame
        Dataframe with encoded binary labels.

    Raises
    ------
    ValueError
        If required source column is missing
        or unknown sources are encountered.
    """

    # ======================
    # COPY DATAFRAME
    # ======================

    if not inplace:
        df = df.copy()

    # ======================
    # SOURCE COLUMN VALIDATION
    # ======================

    if source_col not in df.columns:
        raise ValueError(
            f"Missing required source column: "
            f"'{source_col}'"
        )

    # ======================
    # NORMALIZE SOURCES
    # ======================

    df = normalize_source_column(
        df,
        source_col=source_col
    )

    # ======================
    # VALIDATE KNOWN SOURCES
    # ======================

    observed_sources = set(
        df[source_col].unique()
    )

    unknown_sources = (
        observed_sources - KNOWN_SOURCES
    )

    if unknown_sources:
        raise ValueError(
            "Unknown sources encountered during "
            "label encoding: "
            f"{sorted(unknown_sources)}"
        )

    # ======================
    # ENCODE LABELS
    # ======================

    df[label_col] = df[source_col].apply(
        lambda source:
        0 if source in LEFT_SOURCES else 1
    )

    # ======================
    # FINAL VALIDATION
    # ======================

    if df[label_col].isnull().any():
        raise ValueError(
            "Label encoding produced null values."
        )

    return df

# ======================
# FULL-TEXT DATASET LOADING
# ======================

def load_fulltext_dataset(
    filepath,
    text_column="text",
    label_column="label",
    source_name=None,
    lowercase=DEFAULT_LOWERCASE,
    strip_whitespace=DEFAULT_STRIP_WHITESPACE,
    drop_empty_text=DEFAULT_DROP_EMPTY_TEXT
):
    """
    Load and prepare a full-text dataset from a JSON file
    for article-level political classification experiments.

    Processing stages:
    1. Load raw JSON dataset
    2. Standardize schema/column names
    3. Delegate preprocessing to load_external_dataframe()

    Parameters
    ----------
    filepath : str
        Path to dataset JSON file.

    text_column : str, default="text"
        Expected text column before normalization.

    label_column : str, default="label"
        Expected label column before normalization.

    source_name : str or None, optional
        Fallback source name if source metadata
        is missing from dataset.

    lowercase : bool, default=False
        Whether to lowercase article text.

    strip_whitespace : bool, default=True
        Whether to strip surrounding whitespace.

    drop_empty_text : bool, default=True
        Whether to remove rows with empty text.

    Returns
    -------
    pandas.DataFrame
        Standardized full-text dataframe ready for
        feature extraction and model training.
    """

    # ======================
    # LOAD RAW DATA
    # ======================

    df = load_json_file(filepath)

    if df.empty:
        raise ValueError(
            "Loaded dataset is empty."
        )

    # ======================
    # STANDARDIZE SCHEMA
    # ======================

    df = standardize_columns(df)

    # ======================
    # DELEGATE TO EXTERNAL PIPELINE
    # ======================

    return load_external_dataframe(
        df=df,
        text_col=text_column,
        label_col=label_column,
        source_col="outlet_name",
        source_name=source_name or "external",
        lowercase=lowercase,
        strip_whitespace=strip_whitespace,
        drop_empty_text=drop_empty_text
    )


# ======================
# EXTERNAL DATAFRAME LOADING
# ======================

def load_external_dataframe(
    df,
    text_col="text",
    label_col="label",
    source_col=None,
    source_name="external",
    lowercase=DEFAULT_LOWERCASE,
    strip_whitespace=DEFAULT_STRIP_WHITESPACE,
    drop_empty_text=DEFAULT_DROP_EMPTY_TEXT
):
    """
    Standardize and prepare an externally supplied
    full-text dataset.

    This function supports trainable NLP datasets such as:
    - AG News
    - custom news corpora
    - reconstructed article datasets
    - CSV/JSON imports
    - externally scraped corpora

    The resulting dataframe is normalized into the
    project's standard schema for downstream feature
    extraction, cross-validation, and classification.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing article data.

    text_col : str, default="text"
        Column containing article body text.

    label_col : str, default="label"
        Column containing classification labels.

    source_col : str or None, default=None
        Optional source/outlet column.

    source_name : str, default="external"
        Default source value if no source column exists.

    lowercase : bool, default=False
        Whether to lowercase article text.

    strip_whitespace : bool, default=True
        Whether to trim surrounding whitespace.

    drop_empty_text : bool, default=True
        Whether to remove rows with empty text.

    Returns
    -------
    pandas.DataFrame
        Standardized dataframe ready for training.

    Raises
    ------
    ValueError
        If required columns are missing
        or dataset becomes invalid.
    """

    # ======================
    # BASIC VALIDATION
    # ======================

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    df = df.copy()

    # ======================
    # STANDARDIZE SCHEMA
    # ======================

    df = standardize_columns(df)

    # ======================
    # VALIDATE REQUIRED COLUMNS
    # ======================

    missing_columns = []

    if text_col not in df.columns:
        missing_columns.append(text_col)

    if label_col not in df.columns:
        missing_columns.append(label_col)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # ======================
    # STANDARDIZE CORE COLUMN NAMES
    # ======================

    rename_map = {}

    if text_col != "text":
        rename_map[text_col] = "text"

    if label_col != "label":
        rename_map[label_col] = "label"

    if (
        source_col
        and source_col in df.columns
        and source_col != "outlet_name"
    ):
        rename_map[source_col] = "outlet_name"

    if rename_map:
        df = df.rename(columns=rename_map)

    # ======================
    # ENSURE SOURCE METADATA
    # ======================

    df = ensure_source_column(
        df,
        source_col="outlet_name",
        default_source=source_name
    )

    # ======================
    # NORMALIZE TEXT
    # ======================

    df = normalize_text_column(
        df,
        text_column="text",
        lowercase=lowercase,
        strip_whitespace=strip_whitespace
    )

    # ======================
    # REMOVE EMPTY TEXT
    # ======================

    if drop_empty_text:

        df = remove_empty_text_rows(
            df,
            text_column="text",
            verbose=True
        )

    # ======================
    # KEEP STANDARD COLUMNS
    # ======================

    df = keep_standard_columns(
        df,
        prioritize=[
            "text",
            "label",
            "outlet_name"
        ]
    )

    # ======================
    # FINAL VALIDATION
    # ======================

    df = validate_text_dataset(
        df,
        text_column="text",
        label_column="label"
    )

    # ======================
    # FINAL CLEANUP
    # ======================

    df = df.reset_index(drop=True)

    return df