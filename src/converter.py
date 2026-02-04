"""Main converter logic for transforming EDL and source lists into a definitive archive list."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from .timecode import Timecode
from .models import EDLEntry, SourceEntry, DefEntry


# Encodings to try when reading files
ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']


def read_file_with_encoding(filepath: Path) -> tuple[str, str]:
    """Read a file trying multiple encodings.

    Args:
        filepath: Path to the file

    Returns:
        Tuple of (content, encoding_used)

    Raises:
        UnicodeDecodeError: If no encoding works
    """
    for encoding in ENCODINGS:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            return content, encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        'all', b'', 0, 1,
        f"Could not decode file with any of: {ENCODINGS}"
    )


# Column name mappings: English -> internal name
EDL_COLUMN_MAP = {
    # English names (from AVID/Premiere)
    "id": "id",
    "reel": "reel",
    "name": "name",
    "file name": "file_name",
    "track": "track",
    "timecode in": "timecode_in",
    "timecode out": "timecode_out",
    "duration": "duration",
    "source start": "source_start",
    "source end": "source_end",
    "audio channels": "audio_channels",
    "comment": "comment",
}

# Dutch alternative column names for EDL
EDL_DUTCH_MAP = {
    "bestandsnaam": "name",
    "duur": "duration",
    "tc in": "timecode_in",
    "tc uit": "timecode_out",
    "bron start": "source_start",
    "bron einde": "source_end",
}

# Source list column mappings
SOURCE_COLUMN_MAP = {
    # Dutch names (common in Dutch production)
    "bestandsnaam": "name",
    "omschrijving": "description",
    "link": "link",
    "bron": "source",
    "kosten": "cost",
    "rechten / contact": "rights_contact",
    "to do/opmerkinen": "todo_notes",
    "to do/opmerkingen": "todo_notes",  # Alternative spelling
    "to do": "todo_notes",
    "bron in beeld": "source_in_frame",
    "aftiteling": "credits",
    # Also for DEF format (slightly different column names)
    "tc in": "timecode_in",
    "duur": "duration",
    "prijs nl": "cost",
}

# English source column names
SOURCE_ENGLISH_MAP = {
    "name": "name",
    "description": "description",
    "link": "link",
    "source": "source",
    "cost": "cost",
    "rights_contact": "rights_contact",
    "todo_notes": "todo_notes",
    "source_in_frame": "source_in_frame",
    "credits": "credits",
}


def normalize_column_name(col: str) -> str:
    """Normalize a column name for matching.

    Args:
        col: Original column name

    Returns:
        Normalized lowercase column name
    """
    return col.strip().lower()


def map_columns(df: pd.DataFrame, column_maps: list[dict[str, str]]) -> pd.DataFrame:
    """Map DataFrame columns to internal names using provided mappings.

    Args:
        df: DataFrame with original column names
        column_maps: List of column name mappings to try (in order)

    Returns:
        DataFrame with renamed columns
    """
    # Create a combined mapping from all provided maps
    combined_map = {}
    for cmap in column_maps:
        combined_map.update(cmap)

    # Normalize column names and find matches
    rename_map = {}
    for col in df.columns:
        normalized = normalize_column_name(col)
        if normalized in combined_map:
            rename_map[col] = combined_map[normalized]

    return df.rename(columns=rename_map)


def load_edl(filepath: Path | str, fps: int = 25) -> list[EDLEntry]:
    """Load an EDL from a CSV file.

    Args:
        filepath: Path to the EDL CSV file
        fps: Frame rate for timecode parsing

    Returns:
        List of EDLEntry objects
    """
    filepath = Path(filepath)

    # Read file with encoding detection
    content, encoding = read_file_with_encoding(filepath)
    first_line = content.split('\n')[0] if content else ''

    delimiter = '\t' if '\t' in first_line else ','

    df = pd.read_csv(filepath, delimiter=delimiter, dtype=str, encoding=encoding)
    df = df.fillna("")

    # Map columns
    df = map_columns(df, [EDL_COLUMN_MAP, EDL_DUTCH_MAP])

    entries = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()

        # Skip rows without a name
        name = row_dict.get("name", "")
        if not name or name.strip() == "":
            continue

        try:
            entry = EDLEntry.from_dict(row_dict, fps=fps)
            entries.append(entry)
        except ValueError as e:
            print(f"Warning: Skipping row due to error: {e}")
            continue

    return entries


def load_source(filepath: Path | str) -> list[SourceEntry]:
    """Load a source list from a CSV file.

    Args:
        filepath: Path to the source CSV file

    Returns:
        List of SourceEntry objects
    """
    filepath = Path(filepath)

    # Read file with encoding detection
    content, encoding = read_file_with_encoding(filepath)
    first_line = content.split('\n')[0] if content else ''

    delimiter = '\t' if '\t' in first_line else ','

    df = pd.read_csv(filepath, delimiter=delimiter, dtype=str, encoding=encoding)
    df = df.fillna("")

    # Map columns
    df = map_columns(df, [SOURCE_COLUMN_MAP, SOURCE_ENGLISH_MAP])

    entries = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()

        # Skip rows without a name
        name = row_dict.get("name", "")
        if not name or name.strip() == "":
            continue

        entry = SourceEntry.from_dict(row_dict)
        entries.append(entry)

    return entries


def normalize_name(name: str) -> str:
    """Normalize a file name for matching.

    Removes file extensions and normalizes case and whitespace.

    Args:
        name: Original file name

    Returns:
        Normalized name for matching
    """
    name = name.strip()
    # Remove common extensions
    for ext in ['.mxf', '.mp4', '.mov', '.avi', '.sync.01', '.sync.02', '.sync']:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
    return name.lower().strip()


def collapse_edl(entries: list[EDLEntry], fps: int = 25) -> list[EDLEntry]:
    """Collapse consecutive EDL entries with the same name.

    When two or more consecutive entries have the same name, they are combined
    into a single entry with:
    - timecode_in from the first entry
    - timecode_out from the last entry
    - Combined duration
    - source_start as the minimum of all source_starts
    - source_end as the maximum of all source_ends

    Args:
        entries: List of EDL entries
        fps: Frame rate for timecode operations

    Returns:
        List of collapsed EDL entries
    """
    if not entries:
        return []

    collapsed = []
    i = 0

    while i < len(entries):
        current = entries[i]

        # Look ahead for consecutive entries with the same name
        j = i + 1
        while j < len(entries) and entries[j].name == current.name:
            j += 1

        # If we found consecutive duplicates
        if j > i + 1:
            # Combine all entries from i to j-1
            group = entries[i:j]

            # Calculate combined values
            combined_duration = Timecode.from_frames(0, fps)
            min_source_start = group[0].source_start
            max_source_end = group[0].source_end

            for entry in group:
                combined_duration = combined_duration + entry.duration
                if entry.source_start < min_source_start:
                    min_source_start = entry.source_start
                if entry.source_end > max_source_end:
                    max_source_end = entry.source_end

            # Create combined entry
            combined = EDLEntry(
                id=current.id,
                name=current.name,
                timecode_in=group[0].timecode_in,
                timecode_out=group[-1].timecode_out,
                duration=combined_duration,
                source_start=min_source_start,
                source_end=max_source_end,
                reel=current.reel,
                file_name=current.file_name,
                track=current.track,
                audio_channels=current.audio_channels,
                comment=current.comment,
            )
            collapsed.append(combined)
        else:
            # No consecutive duplicate, keep as is
            collapsed.append(current)

        i = j

    return collapsed


def find_source_match(
    edl_name: str,
    sources: list[SourceEntry]
) -> Optional[SourceEntry]:
    """Find a matching source entry for an EDL entry name.

    Uses normalized name matching to handle file extensions and case differences.

    Args:
        edl_name: Name from the EDL entry
        sources: List of source entries to search

    Returns:
        Matching SourceEntry or None if not found
    """
    normalized_edl = normalize_name(edl_name)

    for source in sources:
        normalized_source = normalize_name(source.name)

        # Exact match after normalization
        if normalized_edl == normalized_source:
            return source

        # Check if one contains the other (for partial matches)
        if normalized_edl in normalized_source or normalized_source in normalized_edl:
            return source

    return None


def generate_def_list(
    edl_entries: list[EDLEntry],
    source_entries: list[SourceEntry]
) -> list[DefEntry]:
    """Generate the definitive archive list from EDL and source entries.

    For each EDL entry, finds the matching source entry and combines them.
    If no source is found, the DEF entry has empty metadata fields.

    Args:
        edl_entries: List of (collapsed) EDL entries
        source_entries: List of source entries

    Returns:
        List of DefEntry objects
    """
    def_list = []

    for edl in edl_entries:
        source = find_source_match(edl.name, source_entries)
        def_entry = DefEntry.from_edl_and_source(edl, source)
        def_list.append(def_entry)

    return def_list


def save_def_list(
    def_list: list[DefEntry],
    output_path: Path | str,
    delimiter: str = '\t'
) -> None:
    """Save the definitive list to a CSV file.

    Args:
        def_list: List of DefEntry objects to save
        output_path: Path for the output file
        delimiter: CSV delimiter (default is tab)
    """
    output_path = Path(output_path)

    # Convert to list of dicts
    rows = [entry.to_dict() for entry in def_list]

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep=delimiter, index=False)


def convert(
    edl_path: Path | str,
    source_path: Path | str,
    output_path: Path | str,
    fps: int = 25,
    collapse: bool = True,
    delimiter: str = '\t'
) -> list[DefEntry]:
    """Main conversion function: EDL + Source -> Definitive List.

    Args:
        edl_path: Path to the EDL CSV file
        source_path: Path to the source list CSV file
        output_path: Path for the output DEF list
        fps: Frame rate for timecode handling
        collapse: Whether to collapse consecutive same-name entries
        delimiter: Output file delimiter

    Returns:
        List of DefEntry objects that were saved
    """
    # Load files
    edl_entries = load_edl(edl_path, fps=fps)
    source_entries = load_source(source_path)

    print(f"Loaded {len(edl_entries)} EDL entries")
    print(f"Loaded {len(source_entries)} source entries")

    # Collapse EDL if requested
    if collapse:
        edl_entries = collapse_edl(edl_entries, fps=fps)
        print(f"Collapsed to {len(edl_entries)} entries")

    # Generate DEF list
    def_list = generate_def_list(edl_entries, source_entries)

    # Count matches
    matched = sum(1 for d in def_list if d.link)
    print(f"Matched {matched}/{len(def_list)} entries with sources")

    # Save output
    save_def_list(def_list, output_path, delimiter=delimiter)
    print(f"Saved to {output_path}")

    return def_list
