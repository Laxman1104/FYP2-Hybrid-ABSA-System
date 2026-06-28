import re
import html
import unicodedata


def standardize_text(text):
    """
    Standardises raw text for downstream NLP processing.
    Does NOT remove stop-words or lemmatize — those steps happen
    immediately before vectorisation in the model training pipeline.
    """
    if not isinstance(text, str):
        return ""

    # 1. Decode HTML entities (e.g. &amp; → &)
    text = html.unescape(text)

    # 2. Normalise unicode (e.g. smart quotes → standard quotes)
    text = unicodedata.normalize('NFKC', text)

    # 3. Lowercase
    text = text.lower()

    # 4. Collapse multiple whitespace into single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def standardize_dataframe_aspects(df, categoriser, label="Dataset"):
    """
    Maps aspect_clean column through AspectCategoriser to produce macro_aspect.
    Drops rows where aspect falls below similarity threshold (classified as noise).
    Reports drop count for transparency.
    """

    if 'aspect_clean' not in df.columns:
        raise ValueError("DataFrame must contain 'aspect_clean' column. "
                     "Run standardize_text() on the aspect column first.")
                     
    processed_df = df.copy()

    # Build mapping dictionary from unique aspects only (efficiency)
    mapping_dict = {}
    for aspect in processed_df['aspect_clean'].unique():
        result = categoriser.categorise([(aspect, "dummy")])
        mapping_dict[aspect] = result[0][0] if result else "DROP"

    processed_df['macro_aspect'] = processed_df['aspect_clean'].map(mapping_dict)

    # Report dropped rows
    total   = len(processed_df)
    dropped = (processed_df['macro_aspect'] == "DROP").sum()
    kept    = total - dropped
    print(f"{label}: {total} rows → dropped {dropped} ({dropped/total*100:.1f}%) → kept {kept} ({kept/total*100:.1f}%)")

    # Drop noise rows
    processed_df = processed_df[processed_df['macro_aspect'] != "DROP"].reset_index(drop=True)

    return processed_df