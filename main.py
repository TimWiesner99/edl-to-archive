#!/usr/bin/env python3
"""EDL to Archive List Converter.

Converts Edit Decision Lists (EDL) from AVID/Premiere Pro and source archive
lists from image researchers into a definitive archive list.

Usage:
    python main.py <edl_file> <source_file> -o <output_file>
    python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv
    python main.py   # Interactive mode: creates template files for you to fill in
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from src.converter import convert, validate_edl_file, validate_source_file
from src.exclusion import load_exclusion_rules, ExclusionRuleSyntaxError


# Default paths for interactive mode
DEFAULT_INPUT_DIR = Path("input")
DEFAULT_EDL_PATH = DEFAULT_INPUT_DIR / "EDL.csv"
DEFAULT_SOURCE_PATH = DEFAULT_INPUT_DIR / "SOURCE.csv"
DEFAULT_OUTPUT_PATH = Path("output") / "DEF.csv"

# Template headers (comma-delimited CSV format)
EDL_TEMPLATE_HEADER = "ID,Reel,Name,File Name,Track,Timecode In,Timecode Out,Duration,Source Start,Source End,Audio Channels,Comment\n"
SOURCE_TEMPLATE_HEADER = "TC in,Duur,Bestandsnaam,Omschrijving,Link,Bron,kosten,rechten / contact,to do,Bron in beeld,Aftiteling\n"


def open_file_in_default_app(filepath: Path) -> bool:
    """Open a file in the OS default application.

    Args:
        filepath: Path to the file to open

    Returns:
        True if the file was opened successfully, False otherwise
    """
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", str(filepath)])
        elif system == "Windows":
            subprocess.Popen(["start", "", str(filepath)], shell=True)
        else:
            subprocess.Popen(["xdg-open", str(filepath)])
        return True
    except (OSError, FileNotFoundError):
        return False


def create_template_file(filepath: Path, header: str) -> bool:
    """Create a template CSV file with only the header row.

    Only creates the file if it doesn't exist or is empty.

    Args:
        filepath: Path for the template file
        header: Header row content

    Returns:
        True if a new template was created, False if file already had content
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # If file exists and has content beyond the header, don't overwrite
    if filepath.exists():
        # Try UTF-8 first, fall back to Windows-1252 (common on Windows)
        for encoding in ("utf-8", "cp1252"):
            try:
                content = filepath.read_text(encoding=encoding).strip()
                break
            except UnicodeDecodeError:
                continue
        else:
            # If all encodings fail, read with error replacement
            content = filepath.read_text(encoding="utf-8", errors="replace").strip()

        if content and content != header.strip():
            return False

    filepath.write_text(header, encoding="utf-8")
    return True


def interactive_input_flow(edl_path: Path, source_path: Path, fps: int) -> bool:
    """Handle the interactive input flow: create templates, open files, wait, validate.

    Args:
        edl_path: Path for the EDL file
        source_path: Path for the source file
        fps: Frame rate for EDL validation

    Returns:
        True if input is valid and ready to proceed, False otherwise
    """
    print("\nNo input files provided. Setting up interactive input...\n")

    # Create template files
    edl_created = create_template_file(edl_path, EDL_TEMPLATE_HEADER)
    source_created = create_template_file(source_path, SOURCE_TEMPLATE_HEADER)

    if edl_created:
        print(f"  Created EDL template:    {edl_path}")
    else:
        print(f"  EDL file already exists:  {edl_path}")

    if source_created:
        print(f"  Created SOURCE template: {source_path}")
    else:
        print(f"  SOURCE file already exists: {source_path}")

    # Open files
    print("\nOpening files for editing...")
    edl_opened = open_file_in_default_app(edl_path)
    source_opened = open_file_in_default_app(source_path)

    if not edl_opened or not source_opened:
        print("  Could not open files automatically.")
        print(f"  Please open them manually:")
        print(f"    EDL:    {edl_path.resolve()}")
        print(f"    SOURCE: {source_path.resolve()}")

    print("\nPaste your data into the files and save them.")

    # Wait for user
    while True:
        print()
        response = input("Waiting for input... Continue? [y/n] ").strip().lower()
        if response == "n":
            print("Aborted.")
            return False
        if response == "y":
            break
        print("Please enter 'y' to continue or 'n' to abort.")

    # Validate input files
    print("\nValidating input files...")
    valid = True

    edl_errors = validate_edl_file(edl_path, fps=fps)
    if edl_errors:
        print(f"\n  EDL file errors ({edl_path}):")
        for err in edl_errors:
            print(f"    - {err}")
        valid = False
    else:
        print(f"  EDL file OK: {edl_path}")

    source_errors = validate_source_file(source_path)
    if source_errors:
        print(f"\n  SOURCE file errors ({source_path}):")
        for err in source_errors:
            print(f"    - {err}")
        valid = False
    else:
        print(f"  SOURCE file OK: {source_path}")

    if not valid:
        print("\nInput validation failed. Please fix the errors and try again.")
        return False

    print()
    return True


def main() -> int:
    """Main entry point for the EDL to Archive converter CLI."""
    parser = argparse.ArgumentParser(
        description="Convert EDL and source archive lists into a definitive archive list.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode (creates template files, opens them for editing)
    python main.py

    # Basic conversion with default settings (25 fps)
    python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv

    # Convert with different frame rate
    python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv --fps 24

    # Without collapsing consecutive entries
    python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv --no-collapse
        """
    )

    parser.add_argument(
        "edl",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the EDL CSV file (from AVID/Premiere Pro). "
             "If omitted, interactive mode is used."
    )

    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the source archive list CSV file (from image researcher). "
             "If omitted, interactive mode is used."
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path for the output definitive list CSV file (default: output/DEF.csv)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frame rate for timecode calculations (default: 25)"
    )

    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="Don't collapse consecutive entries with the same name"
    )

    parser.add_argument(
        "--delimiter",
        choices=["tab", "comma"],
        default="comma",
        help="Output file delimiter (default: comma)"
    )

    parser.add_argument(
        "--exclude",
        type=Path,
        help="Path to exclusion rules file"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbosity: -v for basic output, -vv for detailed evaluation traces"
    )

    parser.add_argument(
        "--frames",
        action="store_true",
        help="Include frame-level precision in timecodes (default: round to seconds)"
    )

    args = parser.parse_args()

    # Determine whether to use interactive mode
    interactive = args.edl is None or args.source is None

    if interactive:
        edl_path = args.edl or DEFAULT_EDL_PATH
        source_path = args.source or DEFAULT_SOURCE_PATH
        output_path = args.output or DEFAULT_OUTPUT_PATH

        if not interactive_input_flow(edl_path, source_path, fps=args.fps):
            return 1
    else:
        edl_path = args.edl
        source_path = args.source
        output_path = args.output or DEFAULT_OUTPUT_PATH

        # Validate input files exist
        if not edl_path.exists():
            print(f"Error: EDL file not found: {edl_path}", file=sys.stderr)
            return 1

        if not source_path.exists():
            print(f"Error: Source file not found: {source_path}", file=sys.stderr)
            return 1

        # Validate input format
        print("Validating input files...")
        edl_errors = validate_edl_file(edl_path, fps=args.fps)
        if edl_errors:
            print(f"\n  EDL file errors ({edl_path}):", file=sys.stderr)
            for err in edl_errors:
                print(f"    - {err}", file=sys.stderr)
            return 1

        source_errors = validate_source_file(source_path)
        if source_errors:
            print(f"\n  SOURCE file errors ({source_path}):", file=sys.stderr)
            for err in source_errors:
                print(f"    - {err}", file=sys.stderr)
            return 1

        print("  Input files OK.")

    if args.exclude and not args.exclude.exists():
        print(f"Error: Exclusion rules file not found: {args.exclude}", file=sys.stderr)
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine delimiter
    delimiter = '\t' if args.delimiter == "tab" else ','

    # Load exclusion rules if provided
    exclusion_rules = None
    if args.exclude:
        try:
            print(f"\nLoading exclusion rules from: {args.exclude}")
            exclusion_rules = load_exclusion_rules(args.exclude)
            print(f"  Loaded {len(exclusion_rules)} exclusion rules\n")
        except ExclusionRuleSyntaxError as e:
            print(f"Error in exclusion rules file: {e}", file=sys.stderr)
            return 1

    # Run conversion
    try:
        print(f"Converting EDL: {edl_path}")
        print(f"With source: {source_path}")
        print(f"Frame rate: {args.fps} fps")
        print(f"Collapse consecutive: {not args.no_collapse}")
        if args.verbose > 0:
            verbose_mode = "basic (-v)" if args.verbose == 1 else "detailed (-vv)" if args.verbose == 2 else f"level {args.verbose}"
            print(f"Verbose mode: {verbose_mode}")
        print("-" * 40)

        convert(
            edl_path=edl_path,
            source_path=source_path,
            output_path=output_path,
            fps=args.fps,
            collapse=not args.no_collapse,
            delimiter=delimiter,
            exclusion_rules=exclusion_rules,
            verbose=args.verbose > 0,
            verbose_level=args.verbose,
            include_frames=args.frames
        )

        print("-" * 40)
        print("Conversion complete!")

        # Open the Excel output file
        excel_path = output_path.parent / f"{output_path.stem}.xlsx"
        if excel_path.exists():
            print(f"\nOpening {excel_path}...")
            open_file_in_default_app(excel_path)

        return 0

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
