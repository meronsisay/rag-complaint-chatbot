"""
Data processing module for CFPB complaint data.
Contains all preprocessing logic, text cleaning, and configuration.
"""

import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA_PATH = "../data/raw/complaints.csv"
FILTERED_DATA_PATH = "../data/processed/filtered_complaints.csv"

# Exact strings from CFPB dataset
TARGET_PRODUCTS = [
    "Credit card",
    "Credit card or prepaid card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Payday loan, title loan, personal loan, or advance loan",
]

# Map to clean categories
CATEGORY_MAP = {
    "Credit card": "Credit Cards",
    "Credit card or prepaid card": "Credit Cards",
    "Checking or savings account": "Savings Accounts",
    "Money transfer, virtual currency, or money service": "Money Transfers",
    "Payday loan, title loan, personal loan, or advance loan": "Personal Loans",
}

# Get unique categories
PRODUCT_CATEGORIES = list(set(CATEGORY_MAP.values()))

# FIXED: Use the actual column name from CSV
NARRATIVE_COL = "Consumer complaint narrative"
CHUNK_SIZE = 100000
BOILERPLATE_PHRASES = [
    # General complaint starters
    r"i am writing to file a complaint",
    r"i am writing to complain about",
    r"i would like to file a complaint",
    r"this is a complaint about",
    r"i am submitting this complaint",
    r"i wish to file a complaint",
    r"please accept this as a complaint",
    r"i am filing a complaint",
    r"this complaint concerns",
    r"i am writing to bring to your attention",
    # Common opening phrases
    r"to whom it may concern",
    r"dear cfpb",
    r"dear consumer financial protection bureau",
    r"dear sir or madam",
    r"dear sirs",
    r"good day",
    # Request/action phrases
    r"i would like to request",
    r"i am requesting",
    r"i am asking",
    r"please investigate",
    r"please look into",
    # Closing phrases
    r"thank you for your time",
    r"thank you for your attention",
    r"thank you in advance",
    r"sincerely",
    r"respectfully",
    r"regards",
    r"best regards",
    r"cordially",
    # Common phrases in financial complaints
    r"i was charged",
    r"i was billed",
    r"i was overcharged",
    r"i was scammed",
    r"i was defrauded",
    r"i was misled",
    r"i was promised",
    r"i was told",
    r"i was informed",
    r"i was notified",
    r"i was unaware",
    r"i was not informed",
    r"i was not notified",
    r"i was not aware",
    # Repetitive phrases
    r"this is a",
    r"this is an",
    r"in regards to",
    r"in reference to",
    r"with regards to",
    r"with reference to",
]

OUTPUT_COLUMNS = [
    "Complaint ID",
    "Product_Category",  # Now using clean category
    "Product",  # Keep original for reference
    "Issue",
    "Sub-issue",
    "Company",
    "State",
    "Date received",
    "narrative",
]


# ============================================================================
# TEXT CLEANING FUNCTIONS
# ============================================================================


def clean_text(text):
    """
    Clean complaint narrative text.

    Args:
        text (str): Raw complaint narrative

    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove special characters (keep letters, numbers, spaces, and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]', " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove boilerplate phrases
    for phrase in BOILERPLATE_PHRASES:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE)

    # Remove leading/trailing punctuation
    text = text.strip(".,!? ")

    return text


def add_word_count(df, text_column="narrative"):
    """Add word count column to dataframe."""
    df["word_count"] = df[text_column].astype(str).str.split().str.len()
    return df


def remove_empty_narratives(df, text_column="narrative"):
    """Remove rows with empty narratives."""
    return df[df[text_column].str.len() > 0].copy()


# ============================================================================
# MAIN DATA PROCESSOR CLASS
# ============================================================================


class ComplaintDataProcessor:
    """
    Main class for processing CFPB complaint data.
    Handles loading, filtering, cleaning, and saving.
    """

    def __init__(self, raw_path=RAW_DATA_PATH, filtered_path=FILTERED_DATA_PATH):
        """
        Initialize the processor.

        Args:
            raw_path (str): Path to raw data
            filtered_path (str): Path to save filtered data
        """
        self.raw_path = raw_path
        self.filtered_path = filtered_path
        self.full_df = None
        self.filtered_df = None
        self.cleaned_df = None

    # ------------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------------

    def load_and_filter_dataset(self, chunksize=CHUNK_SIZE):
        """
        Load and filter the dataset in one pass to save memory.
        This is the main method for processing the large file.
        """
        print(f"Processing {self.raw_path} in chunks...")
        print(f"Chunk size: {chunksize:,} rows")
        print("-" * 60)

        filtered_chunks = []
        chunk_count = 0
        total_processed = 0
        total_kept = 0

        # Read CSV in chunks
        reader = pd.read_csv(self.raw_path, chunksize=chunksize, low_memory=False)

        for chunk in reader:
            chunk_count += 1
            total_processed += len(chunk)

            # Filter immediately: only keep target products
            chunk_filtered = chunk[chunk["Product"].isin(TARGET_PRODUCTS)].copy()
            kept_in_chunk = len(chunk_filtered)

            if kept_in_chunk > 0:
                # NEW: Map to clean categories
                chunk_filtered["Product_Category"] = chunk_filtered["Product"].map(
                    CATEGORY_MAP
                )
                filtered_chunks.append(chunk_filtered)
                total_kept += kept_in_chunk

            # Progress update - only every 10 chunks
            if chunk_count % 10 == 0:
                print(
                    f"  Processed {total_processed:,} rows... (kept {total_kept:,} so far)"
                )

        print("-" * 60)
        print(f"Processed {total_processed:,} rows across {chunk_count} chunks")
        print(f"Kept {total_kept:,} records for target products")

        # Concatenate only the filtered data
        if filtered_chunks:
            self.filtered_df = pd.concat(filtered_chunks, ignore_index=True)
            print(f"Final filtered dataset: {len(self.filtered_df):,} records")

            # NEW: Show category breakdown
            print("\nCategory Breakdown:")
            for category, count in (
                self.filtered_df["Product_Category"].value_counts().items()
            ):
                pct = count / len(self.filtered_df) * 100
                print(f"  {category}: {count:,} ({pct:.1f}%)")

            return self.filtered_df
        else:
            print("✗ No records matched the filter criteria.")
            return pd.DataFrame()

    # ------------------------------------------------------------------------
    # CLEANING
    # ------------------------------------------------------------------------

    def clean_narratives(self, df=None, narrative_col=NARRATIVE_COL):
        """
        Clean complaint narratives.

        Args:
            df (pd.DataFrame): Input dataframe
            narrative_col (str): Name of narrative column (actual CSV column name)

        Returns:
            pd.DataFrame: Dataframe with cleaned narratives
        """
        if df is None:
            df = self.filtered_df

        if df is None or len(df) == 0:
            print(" No data to clean.")
            return None

        print("\n" + "=" * 60)
        print("CLEANING NARRATIVES")
        print("=" * 60)
        print(f"Starting with {len(df):,} records")

        # Clean narratives using the actual column name
        print("  Applying text cleaning...")
        df["cleaned_narrative"] = df[narrative_col].apply(clean_text)

        # Remove empty narratives
        print("  Removing empty narratives...")
        df_cleaned = remove_empty_narratives(df, "cleaned_narrative")
        print(f"  Removed {len(df) - len(df_cleaned):,} empty records")

        # Add word count
        print("  Adding word count...")
        df_cleaned = add_word_count(df_cleaned, "cleaned_narrative")

        # Rename for consistency
        df_cleaned = df_cleaned.rename(columns={"cleaned_narrative": "narrative"})

        self.cleaned_df = df_cleaned
        print(f"\n Cleaned dataset: {len(self.cleaned_df):,} records")

        # NEW: Show category breakdown after cleaning
        if "Product_Category" in self.cleaned_df.columns:
            print("\nCategory Breakdown (Cleaned):")
            for category, count in (
                self.cleaned_df["Product_Category"].value_counts().items()
            ):
                pct = count / len(self.cleaned_df) * 100
                print(f"  {category}: {count:,} ({pct:.1f}%)")

        return self.cleaned_df

    # ------------------------------------------------------------------------
    # SAVING
    # ------------------------------------------------------------------------

    def save_filtered_data(self, df=None, columns=None, output_path=None):
        """
        Save filtered and cleaned dataset.
        """
        if df is None:
            df = self.cleaned_df

        if df is None or len(df) == 0:
            print(" No data to save.")
            return None

        if columns is None:
            columns = OUTPUT_COLUMNS

        if output_path is None:
            output_path = self.filtered_path

        # Only keep columns that exist
        existing_cols = [col for col in columns if col in df.columns]
        df_to_save = df[existing_cols].copy()

        df_to_save.to_csv(output_path, index=False)
        print(f"\n Saved {len(df_to_save):,} records to {output_path}")
        print(f"Columns saved: {df_to_save.columns.tolist()}")

        return df_to_save

    # ------------------------------------------------------------------------
    # SUMMARY & UTILITIES
    # ------------------------------------------------------------------------

    def get_product_distribution(self, df):
        """Get product distribution as a Series."""
        if "Product_Category" in df.columns:
            return df["Product_Category"].value_counts()
        return df["Product"].value_counts()

    def get_issue_distribution(self, df):
        """Get issue distribution as a Series."""
        return df["Issue"].value_counts()

    def get_company_distribution(self, df):
        """Get company distribution as a Series."""
        return df["Company"].value_counts()

    # ------------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------------

    def run_pipeline(self, save=True):
        """
        Run the complete preprocessing pipeline.
        """
        # Step 1: Load and filter in one pass
        self.filtered_df = self.load_and_filter_dataset()

        # Check if we actually got data back
        if self.filtered_df.empty:
            print("Pipeline aborted: No data found.")
            return None

        # Step 2: Clean narratives
        self.clean_narratives(df=self.filtered_df)

        # Step 3: Save (if requested)
        if save:
            self.save_filtered_data(df=self.cleaned_df)

        return self.cleaned_df
