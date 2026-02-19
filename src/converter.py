"""Main converter logic for transforming EDL and source lists into a definitive archive list."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from .timecode import Timecode
from .models import EDLEntry, SourceEntry, DefEntry
from .exclusion import ExclusionRuleSet, filter_edl_entries


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
    "rechten/contact": "rights_contact",  # Alternative format
    "to do/opmerkinen": "todo_notes",
    "to do/opmerkingen": "todo_notes",  # Alternative spelling
    "to do": "todo_notes",
    "bron in beeld": "source_in_frame",
    "aftiteling": "credits",
    # Also for DEF format (slightly different column names)
    "tc in": "timecode_in",
    "duur": "duration",
    "prijs nl": "price_nl",
    "prijs sales": "price_sales",
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

    Raises:
        ValueError: If validation fails (e.g., conflicting price fields)
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
    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        # Skip rows without a name
        name = row_dict.get("name", "")
        if not name or name.strip() == "":
            continue

        # Row number for error messages (add 2: 1 for header, 1 for 0-indexing)
        row_number = idx + 2

        try:
            entry = SourceEntry.from_dict(row_dict, row_number=row_number)
            entries.append(entry)
        except ValueError as e:
            # Re-raise with file context
            raise ValueError(f"Error in source file '{filepath.name}': {e}") from e

    return entries


def validate_edl_file(filepath: Path | str, fps: int = 25) -> list[str]:
    """Validate an EDL file has correct format and parseable content.

    Checks:
    - File is not empty (beyond headers)
    - Required columns are present after mapping
    - Timecodes are parseable

    Args:
        filepath: Path to the EDL CSV file
        fps: Frame rate for timecode validation

    Returns:
        List of error messages (empty if valid)
    """
    filepath = Path(filepath)
    errors = []

    try:
        content, encoding = read_file_with_encoding(filepath)
    except UnicodeDecodeError:
        return ["Could not read file with any supported encoding."]

    lines = [l for l in content.strip().split('\n') if l.strip()]
    if len(lines) < 1:
        return ["File is empty."]

    first_line = lines[0]
    delimiter = '\t' if '\t' in first_line else ','

    try:
        df = pd.read_csv(filepath, delimiter=delimiter, dtype=str, encoding=encoding)
    except Exception as e:
        return [f"Could not parse CSV: {e}"]

    df = df.fillna("")
    df = map_columns(df, [EDL_COLUMN_MAP, EDL_DUTCH_MAP])

    # Check required columns
    required = ["name", "timecode_in", "timecode_out", "duration", "source_start", "source_end"]
    mapped_cols = set(df.columns)
    missing = [col for col in required if col not in mapped_cols]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        errors.append(f"  Found columns: {', '.join(df.columns.tolist())}")
        return errors

    # Check for data rows
    data_rows = df[df["name"].str.strip() != ""]
    if len(data_rows) == 0:
        errors.append("No data rows found (all 'Name' fields are empty).")
        return errors

    # Validate timecodes on first few rows
    tc_cols = ["timecode_in", "timecode_out", "duration", "source_start", "source_end"]
    for idx, row in data_rows.head(5).iterrows():
        for col in tc_cols:
            val = str(row.get(col, "")).strip()
            if val and val != "00:00:00:00":
                if not Timecode.TIMECODE_PATTERN.match(val):
                    errors.append(f"Row {idx + 2}, column '{col}': invalid timecode format '{val}' (expected HH:MM:SS:FF)")

    return errors


def validate_source_file(filepath: Path | str) -> list[str]:
    """Validate a source file has correct format and content.

    Checks:
    - File is not empty (beyond headers)
    - Required 'name'/'Bestandsnaam' column is present

    Args:
        filepath: Path to the source CSV file

    Returns:
        List of error messages (empty if valid)
    """
    filepath = Path(filepath)
    errors = []

    try:
        content, encoding = read_file_with_encoding(filepath)
    except UnicodeDecodeError:
        return ["Could not read file with any supported encoding."]

    lines = [l for l in content.strip().split('\n') if l.strip()]
    if len(lines) < 1:
        return ["File is empty."]

    first_line = lines[0]
    delimiter = '\t' if '\t' in first_line else ','

    try:
        df = pd.read_csv(filepath, delimiter=delimiter, dtype=str, encoding=encoding)
    except Exception as e:
        return [f"Could not parse CSV: {e}"]

    df = df.fillna("")
    df = map_columns(df, [SOURCE_COLUMN_MAP, SOURCE_ENGLISH_MAP])

    # Check required column
    if "name" not in df.columns:
        errors.append("Missing required column: 'Bestandsnaam' (or 'name')")
        errors.append(f"  Found columns: {', '.join(df.columns.tolist())}")
        return errors

    # Check for data rows
    data_rows = df[df["name"].str.strip() != ""]
    if len(data_rows) == 0:
        errors.append("No data rows found (all 'Bestandsnaam' fields are empty).")
        return errors

    return errors


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


def safe_source_usage(source_start: Timecode, source_end: Timecode, fps: int = 25) -> Timecode:
    """Calculate source usage duration with a minimum of 1 second.

    When source timecodes have been clamped due to framerate mismatch (e.g. 50fps
    source in a 25fps project), source_end can end up less than source_start for
    very short clips within the same second. In that case, return 1 second as the
    minimum duration rather than crashing.

    Args:
        source_start: Source start timecode
        source_end: Source end timecode
        fps: Frame rate

    Returns:
        Duration timecode (minimum 1 second)
    """
    diff_frames = source_end.to_frames() - source_start.to_frames()
    if diff_frames < fps:
        # Less than 1 second (or negative) — use 1 second minimum
        return Timecode.from_frames(fps, fps)
    return Timecode.from_frames(diff_frames, fps)


def collapse_edl(entries: list[EDLEntry], fps: int = 25, verbose: bool = False) -> list[EDLEntry]:
    """Collapse consecutive EDL entries with the same name and continuous source timecodes.

    When two or more consecutive entries have the same name AND their source
    timecodes are continuous (within 1 second), they are combined into a single
    entry with:
    - timecode_in from the first entry
    - timecode_out from the last entry
    - Combined duration
    - source_start as the minimum of all source_starts
    - source_end as the maximum of all source_ends

    Entries with the same name but non-continuous source positions (e.g. different
    parts of the same source clip) are kept separate.

    Args:
        entries: List of EDL entries
        fps: Frame rate for timecode operations
        verbose: If True, print details for each collapse operation

    Returns:
        List of collapsed EDL entries
    """
    if not entries:
        return []

    one_second = Timecode.from_frames(fps, fps)

    collapsed = []
    i = 0

    while i < len(entries):
        current = entries[i]

        # Look ahead for consecutive entries with the same name AND continuous source timecodes
        j = i + 1
        while j < len(entries) and entries[j].name == current.name:
            prev = entries[j - 1]
            curr = entries[j]
            # Check if source timecodes are continuous (within 1 second margin)
            gap_frames = curr.source_start.to_frames() - prev.source_end.to_frames()
            if gap_frames < 0:
                gap_frames = -gap_frames
            if gap_frames > one_second.to_frames():
                break
            j += 1

        # If we found consecutive duplicates with continuous source
        if j > i + 1:
            # Combine all entries from i to j-1
            group = entries[i:j]

            if verbose:
                print(f"  Collapsing {len(group)} consecutive entries: \"{current.name}\"")

            # Calculate combined values
            combined_duration = Timecode.from_frames(0, fps)
            source_usage_sum = Timecode.from_frames(0, fps)
            min_source_start = group[0].source_start
            max_source_end = group[0].source_end

            for entry in group:
                combined_duration = combined_duration + entry.duration
                # Accumulate each individual source usage (preserving actual durations)
                if entry.source_total_usage is not None:
                    source_usage_sum = source_usage_sum + entry.source_total_usage
                else:
                    source_usage_sum = source_usage_sum + safe_source_usage(entry.source_start, entry.source_end, fps)
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
                source_total_usage=source_usage_sum,
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
    source_entries: list[SourceEntry],
    verbose: bool = False
) -> list[DefEntry]:
    """Generate the definitive archive list from EDL and source entries.

    For each EDL entry, finds the matching source entry and combines them.
    If no source is found, the DEF entry has empty metadata fields.

    Args:
        edl_entries: List of (collapsed) EDL entries
        source_entries: List of source entries
        verbose: If True, print details for each match

    Returns:
        List of DefEntry objects
    """
    def_list = []

    for i, edl in enumerate(edl_entries, start=1):
        source = find_source_match(edl.name, source_entries)
        def_entry = DefEntry.from_edl_and_source(edl, source)
        def_list.append(def_entry)

        if verbose:
            if source:
                print(f"  Entry {i}: \"{edl.name}\" -> matched source \"{source.name}\"")
            else:
                print(f"  Entry {i}: \"{edl.name}\" -> NO SOURCE MATCH")

    return def_list


def annotate_occurrences(def_list: list[DefEntry]) -> None:
    """Annotate each DefEntry with its occurrence number and total occurrences.

    After annotation, each entry knows how many times its source appears in total
    and which occurrence it is (1-based). This is used to:
    - Show "Nummer/aantal" (e.g. "2/4") in the output
    - Show cost only on first occurrence, "zie boven" on subsequent

    Modifies entries in place.

    Args:
        def_list: List of DefEntry objects to annotate
    """
    from collections import Counter

    # Count total occurrences of each name
    name_counts = Counter(entry.name for entry in def_list)

    # Track current occurrence number per name
    current_occurrence: dict[str, int] = {}

    for entry in def_list:
        name = entry.name
        current_occurrence[name] = current_occurrence.get(name, 0) + 1
        entry.occurrence_number = current_occurrence[name]
        entry.total_occurrences = name_counts[name]


def print_exclusion_summary(
    excluded: list[EDLEntry],
    rules: ExclusionRuleSet,
    verbose: bool = False
) -> None:
    """Print summary statistics about exclusions.

    Shows total excluded and breakdown by rule.

    Args:
        excluded: List of excluded entries
        rules: The exclusion rule set
        verbose: If True, show detailed breakdown by rule
    """
    if not excluded:
        return

    print(f"\nExclusion Summary:")
    print(f"  Total excluded: {len(excluded)}")

    if verbose:
        stats = rules.get_exclusion_stats(excluded)
        print(f"\n  Breakdown by rule:")
        for rule in rules.rules:
            count = stats.get(rule.line_number, 0)
            if count > 0:
                print(f"    Rule {rule.line_number}: {count} entries")
                print(f"      \"{rule.text}\"")


def save_def_list(
    def_list: list[DefEntry],
    output_path: Path | str,
    delimiter: str = ',',
    include_frames: bool = False
) -> None:
    """Save the definitive list to a CSV file.

    Args:
        def_list: List of DefEntry objects to save
        output_path: Path for the output file
        delimiter: CSV delimiter (default is comma)
        include_frames: If True, include frame-level precision in timecodes
    """
    output_path = Path(output_path)

    # Convert to list of dicts
    rows = [entry.to_dict(include_frames=include_frames) for entry in def_list]

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep=delimiter, index=False)


def read_raw_csv(filepath: Path | str) -> pd.DataFrame:
    """Read a CSV file as-is, preserving original column names and data.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with original column names and string data
    """
    filepath = Path(filepath)
    content, encoding = read_file_with_encoding(filepath)
    first_line = content.split('\n')[0] if content else ''
    delimiter = '\t' if '\t' in first_line else ','
    df = pd.read_csv(filepath, delimiter=delimiter, dtype=str, encoding=encoding)
    df = df.fillna("")
    return df


def save_excel_output(
    edl_path: Path | str,
    source_path: Path | str,
    def_list: list[DefEntry],
    output_path: Path | str,
    include_frames: bool = False
) -> None:
    """Save all data to a single Excel file with three sheets.

    Sheets:
    - SOURCE: Original source archive list
    - EDL: Original edit decision list
    - DEF: Definitive archive list (with occurrence tracking and cost deduplication)

    Args:
        edl_path: Path to the original EDL CSV file
        source_path: Path to the original source CSV file
        def_list: List of DefEntry objects (already annotated with occurrences)
        output_path: Path for the output Excel file (.xlsx)
        include_frames: If True, include frame-level precision in timecodes
    """
    output_path = Path(output_path)

    # Read raw input files
    edl_raw = read_raw_csv(edl_path)
    source_raw = read_raw_csv(source_path)

    # Build DEF DataFrame
    def_rows = [entry.to_dict(include_frames=include_frames) for entry in def_list]
    def_df = pd.DataFrame(def_rows)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        source_raw.to_excel(writer, sheet_name="SOURCE", index=False)
        edl_raw.to_excel(writer, sheet_name="EDL", index=False)
        def_df.to_excel(writer, sheet_name="DEF", index=False)

        workbook = writer.book

        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#B8CCE4',  # Light blue
        })
        currency_format = workbook.add_format({
            'num_format': '€ #,##0',  # Euro, no decimals
        })
        currency_header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#B8CCE4',
            'num_format': '€ #,##0',
        })
        text_format = workbook.add_format({
            'num_format': '@',  # Explicit text format to prevent fraction interpretation
        })

        # Apply header formatting to all sheets
        for sheet_name, df in [
            ("SOURCE", source_raw),
            ("EDL", edl_raw),
            ("DEF", def_df),
        ]:
            worksheet = writer.sheets[sheet_name]
            for col_num, col_name in enumerate(df.columns):
                worksheet.write(0, col_num, col_name, header_format)

        # DEF sheet specific formatting
        def_sheet = writer.sheets["DEF"]

        # Format Kosten column as numbers (skip "zie boven" entries)
        if "Kosten" in def_df.columns:
            kosten_col = def_df.columns.get_loc("Kosten")
            for row_num, value in enumerate(def_df["Kosten"], start=1):
                if value and str(value).strip():
                    str_value = str(value).strip()
                    if str_value == "zie boven":
                        # Keep as text string
                        def_sheet.write_string(row_num, kosten_col, str_value)
                    else:
                        # Parse string to number, removing currency symbols and whitespace
                        clean_value = str_value.replace('€', '').replace(',', '.').strip()
                        try:
                            num_value = float(clean_value)
                            def_sheet.write_number(row_num, kosten_col, num_value, currency_format)
                        except ValueError:
                            # Keep as string if not parseable
                            def_sheet.write(row_num, kosten_col, value)

        # Format Nummer/aantal column as text to prevent "2/4" being interpreted as a date/fraction
        if "Nummer/aantal" in def_df.columns:
            nummer_col = def_df.columns.get_loc("Nummer/aantal")
            for row_num, value in enumerate(def_df["Nummer/aantal"], start=1):
                if value and str(value).strip():
                    def_sheet.write_string(row_num, nummer_col, str(value), text_format)

        # Add Kosten sum to DEF sheet
        if "Kosten" in def_df.columns:
            kosten_col = def_df.columns.get_loc("Kosten")
            data_rows = len(def_df)
            # Sum row: 3 rows below last data (row 0 is header, so last data is at row data_rows)
            sum_row = data_rows + 1 + 3  # +1 for header, +3 for empty rows
            # Excel formula uses 1-based row numbers; data starts at row 2
            sum_formula = f"=SUM({chr(65 + kosten_col)}2:{chr(65 + kosten_col)}{data_rows + 1})"
            # Add "Kosten totaal" label to the left of the sum cell
            def_sheet.write(sum_row, kosten_col - 1, "Kosten totaal", header_format)
            def_sheet.write_formula(sum_row, kosten_col, sum_formula, currency_header_format)


def convert(
    edl_path: Path | str,
    source_path: Path | str,
    output_path: Path | str,
    fps: int = 25,
    collapse: bool = True,
    delimiter: str = ',',
    exclusion_rules: ExclusionRuleSet | None = None,
    verbose: bool = False,
    verbose_level: int = 1,
    include_frames: bool = False
) -> list[DefEntry]:
    """Main conversion function: EDL + Source -> Definitive List.

    Args:
        edl_path: Path to the EDL CSV file
        source_path: Path to the source list CSV file
        output_path: Path for the output DEF list
        fps: Frame rate for timecode handling
        collapse: Whether to collapse consecutive same-name entries
        delimiter: Output file delimiter
        exclusion_rules: Optional exclusion rules to filter entries before processing
        verbose: If True, print detailed progress for each entry
        verbose_level: 1 = basic output, 2 = detailed evaluation traces
        include_frames: If True, include frame-level precision in output timecodes

    Returns:
        List of DefEntry objects that were saved
    """
    # Step 1: Load EDL
    print("Loading EDL file...")
    edl_entries = load_edl(edl_path, fps=fps)
    print(f"  Loaded {len(edl_entries)} EDL entries")

    # Step 2: Load source
    print("Loading source file...")
    source_entries = load_source(source_path)
    print(f"  Loaded {len(source_entries)} source entries")

    # Step 3: Apply exclusion rules (before collapse)
    if exclusion_rules:
        print(f"Applying {len(exclusion_rules)} exclusion rules...")
        edl_entries, excluded = filter_edl_entries(
            edl_entries, exclusion_rules, verbose=verbose, verbose_level=verbose_level
        )
        print(f"  Excluded {len(excluded)} entries, {len(edl_entries)} remaining")

        # Print summary if verbose
        if verbose:
            print_exclusion_summary(excluded, exclusion_rules, verbose=True)

    # Step 4: Collapse EDL if requested
    if collapse:
        print("Collapsing consecutive entries...")
        before_count = len(edl_entries)
        edl_entries = collapse_edl(edl_entries, fps=fps, verbose=verbose)
        print(f"  Collapsed {before_count} entries to {len(edl_entries)}")

    # Step 5: Generate DEF list
    print("Matching EDL entries with sources...")
    def_list = generate_def_list(edl_entries, source_entries, verbose=verbose)
    matched = sum(1 for d in def_list if d.link)
    print(f"  Matched {matched}/{len(def_list)} entries with sources")

    # Step 6: Annotate occurrences (for Nummer/aantal and cost deduplication)
    print("Annotating source occurrences...")
    annotate_occurrences(def_list)

    # Step 7: Save DEF output
    print("Saving output...")
    save_def_list(def_list, output_path, delimiter=delimiter, include_frames=include_frames)
    print(f"  Saved DEF list to {output_path}")

    # Step 8: Save combined Excel output
    output_path = Path(output_path)
    excel_path = output_path.parent / f"{output_path.stem}.xlsx"
    print("Saving Excel output...")
    save_excel_output(
        edl_path=edl_path,
        source_path=source_path,
        def_list=def_list,
        output_path=excel_path,
        include_frames=include_frames,
    )
    print(f"  Saved Excel file to {excel_path}")

    return def_list
