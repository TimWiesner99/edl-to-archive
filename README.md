# edl-to-archive
Simple script to convert EDL and archive source lists to complete archive lists with timecodes and source links.

## Getting Started

### Running the application

**Windows:** Double-click `run.bat`

**macOS / Linux:** Double-click `run.sh` (or run `./run.sh` in a terminal)

The launcher script automatically:
1. Installs [uv](https://docs.astral.sh/uv/) (a fast Python package manager) if not already present
2. Downloads the correct Python version if needed
3. Installs all dependencies
4. Starts the GUI

The first launch may take a moment while dependencies are downloaded. Subsequent launches are near-instant.

### CLI Usage

For command-line usage, run commands through `uv run`:

```bash
uv run python main.py <edl_file> <source_file> -o <output_file> [options]
```

### Options

- `--fps <int>` - Frame rate for timecode calculations (default: 25)
- `--no-collapse` - Don't collapse consecutive entries with the same name
- `--delimiter <tab|comma>` - Output file delimiter (default: comma)
- `--exclude <file>` - Path to exclusion rules file
- `-v, --verbose` - Verbosity levels:
  - `-v` - Basic verbose output (shows which entries are excluded and matched)
  - `-vv` - Detailed verbose output (includes evaluation traces showing why rules matched)

### Examples

```bash
# Basic conversion
uv run python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv

# With exclusion rules
uv run python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv --exclude exclude.txt

# With verbose output to see which entries are excluded
uv run python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv --exclude exclude.txt -v

# With detailed verbose output to debug exclusion rules
uv run python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv --exclude exclude.txt -vv

# Different frame rate, tab-delimited output
uv run python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv --fps 24 --delimiter tab
```

## Exclusion Rules

Exclusion rules allow you to filter out EDL entries before processing. Rules are defined in a text file, one rule per line.

### Syntax

```
field_name OPERATOR "value"
```

**Operators:**
- `IS` - exact match (case-sensitive)
- `INCLUDES` - substring match (case-sensitive)

**Logical operators:**
- `AND` - both conditions must be true
- `OR` - either condition must be true
- `NOT` - negates the following expression
- `()` - parentheses for grouping

Lines are OR'd together: if ANY line matches, the entry is excluded.

### Available Fields

| Field Name | Aliases | Description |
|------------|---------|-------------|
| `name` | `Name`, `NAME` | Clip name |
| `file_name` | `FileName`, `filename`, `Bestandsnaam` | Source file name |
| `reel` | `Reel`, `REEL` | Reel identifier |
| `track` | `Track`, `TRACK` | Track name |
| `comment` | `Comment`, `COMMENT` | Entry comment |

### Example Rules File

```
# Exclude SYNC files with empty file names
file_name IS "" AND name INCLUDES "SYNC"

# Exclude anything marked for exclusion
name INCLUDES "exclude"

# Exclude test entries
name INCLUDES "TEST" OR name INCLUDES "test"

# Complex rule with grouping
(name INCLUDES "TEMP" OR name INCLUDES "temp") AND NOT comment INCLUDES "keep"
```

Lines starting with `#` are treated as comments.

### Debugging with Verbose Output

When troubleshooting exclusion rules, use the `-v` or `-vv` flags:

**Basic verbose (`-v`):** Shows which entries are excluded and which rule matched:
```
Entry 42 EXCLUDED:
  name: "SYNC_VIDEO.MXF"
  file_name: ""
  reel: "001"
  track: "V1"
  comment: ""

  Matched rule 1: file_name IS "" AND name INCLUDES "SYNC"
```

**Detailed verbose (`-vv`):** Also shows the evaluation trace with field values and expression results:
```
Entry 42 EXCLUDED:
  name: "SYNC_VIDEO.MXF"
  file_name: ""
  reel: "001"
  track: "V1"
  comment: ""

  Matched rule 1: file_name IS "" AND name INCLUDES "SYNC"

  Evaluation trace:
    ✓ AND (True AND True -> True)
        ✓ file_name IS ""
           (actual value: "" -> True)
        ✓ name INCLUDES "SYNC"
           (actual value: "SYNC_VIDEO.MXF" -> True)
```

The evaluation trace shows:
- ✓ for conditions that evaluated to true
- ✗ for conditions that evaluated to false
- Actual field values from the entry
- How logical operators (AND/OR/NOT) combined the results
