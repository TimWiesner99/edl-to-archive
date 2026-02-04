# edl-to-archive
Simple script to convert EDL and archive source lists to complete archive lists with timecodes and source links.

## Usage

```bash
python main.py <edl_file> <source_file> -o <output_file> [options]
```

### Options

- `--fps <int>` - Frame rate for timecode calculations (default: 25)
- `--no-collapse` - Don't collapse consecutive entries with the same name
- `--delimiter <tab|comma>` - Output file delimiter (default: comma)
- `--exclude <file>` - Path to exclusion rules file

### Examples

```bash
# Basic conversion
python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv

# With exclusion rules
python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv --exclude exclude.txt

# Different frame rate, tab-delimited output
python main.py input/EDL.csv input/SOURCE.csv -o output/DEF.csv --fps 24 --delimiter tab
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
