#!/usr/bin/env python3
"""EDL to Archive List Converter.

Converts Edit Decision Lists (EDL) from AVID/Premiere Pro and source archive
lists from image researchers into a definitive archive list.

Usage:
    python main.py <edl_file> <source_file> -o <output_file>
    python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv
"""

import argparse
import sys
from pathlib import Path

from src.converter import convert
from src.exclusion import load_exclusion_rules, ExclusionRuleSyntaxError


def main() -> int:
    """Main entry point for the EDL to Archive converter CLI."""
    parser = argparse.ArgumentParser(
        description="Convert EDL and source archive lists into a definitive archive list.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
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
        help="Path to the EDL CSV file (from AVID/Premiere Pro)"
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Path to the source archive list CSV file (from image researcher)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path for the output definitive list CSV file"
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

    # Validate input files exist
    if not args.edl.exists():
        print(f"Error: EDL file not found: {args.edl}", file=sys.stderr)
        return 1

    if not args.source.exists():
        print(f"Error: Source file not found: {args.source}", file=sys.stderr)
        return 1

    if args.exclude and not args.exclude.exists():
        print(f"Error: Exclusion rules file not found: {args.exclude}", file=sys.stderr)
        return 1

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

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
        print(f"Converting EDL: {args.edl}")
        print(f"With source: {args.source}")
        print(f"Frame rate: {args.fps} fps")
        print(f"Collapse consecutive: {not args.no_collapse}")
        if args.verbose > 0:
            verbose_mode = "basic (-v)" if args.verbose == 1 else "detailed (-vv)" if args.verbose == 2 else f"level {args.verbose}"
            print(f"Verbose mode: {verbose_mode}")
        print("-" * 40)

        convert(
            edl_path=args.edl,
            source_path=args.source,
            output_path=args.output,
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
        return 0

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
