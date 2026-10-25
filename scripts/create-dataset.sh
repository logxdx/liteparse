#!/usr/bin/env bash
# Creates a dataset for regression testing
#
# Output structure:
#   dataset/
#     data/
#       doc1.pdf
#       doc2.docx
#       ...
#     metadata.jsonl  (each line: {"file_name":"data/doc1.pdf","document":"doc1.pdf","page":1,"output_text":"...","output_json":{...}})
#
# Usage:
#   ./scripts/create-dataset.sh [output-dir] [source-docs-dir]
#
# Arguments:
#   output-dir      - Where to write the dataset (default: ./dataset)
#   source-docs-dir - Where to read source documents from (default: ./e2e-test-docs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${1:-$REPO_ROOT/dataset}"
SOURCE_DOCS_DIR="${2:-$REPO_ROOT/e2e-test-docs}"
DOCUMENTS_DIR="$OUTPUT_DIR/data"

# Resolve paths for comparison
RESOLVED_SOURCE="$(cd "$SOURCE_DOCS_DIR" 2>/dev/null && pwd)"
RESOLVED_DOCUMENTS="$(mkdir -p "$DOCUMENTS_DIR" && cd "$DOCUMENTS_DIR" && pwd)"
SKIP_COPY=false
if [ "$RESOLVED_SOURCE" = "$RESOLVED_DOCUMENTS" ]; then
  SKIP_COPY=true
fi

LIT="$REPO_ROOT/target/release/liteparse"
if [ ! -x "$LIT" ]; then
  echo "ERROR: liteparse binary not found at $LIT. Run 'cargo build --release' first."
  exit 1
fi

echo "LiteParse Dataset Generator"
echo "==========================="
echo "Source: $SOURCE_DOCS_DIR"
echo "Output: $OUTPUT_DIR"
if [ "$SKIP_COPY" = true ]; then
  echo "(Source is output documents dir - skipping copy)"
fi
echo

mkdir -p "$DOCUMENTS_DIR"

METADATA_FILE="$OUTPUT_DIR/metadata.jsonl"
: > "$METADATA_FILE"

TOTAL_ENTRIES=0
TOTAL_FILES=0

# Find all files in source directory
while IFS= read -r -d '' file; do
  REL_PATH="${file#"$SOURCE_DOCS_DIR/"}"
  TOTAL_FILES=$((TOTAL_FILES + 1))
  echo "Processing: $REL_PATH"

  # Copy file to dataset if needed
  if [ "$SKIP_COPY" = false ]; then
    DEST_PATH="$DOCUMENTS_DIR/$REL_PATH"
    mkdir -p "$(dirname "$DEST_PATH")"
    cp "$file" "$DEST_PATH"
  fi

  # Parse with liteparse and capture JSON output to a temp file (avoids argument list too long)
  JSON_TMPFILE=$(mktemp)
  trap "rm -f '$JSON_TMPFILE'" EXIT
  PARSE_ERROR=""
  if "$LIT" parse --format json --no-ocr -q "$file" > "$JSON_TMPFILE" 2>&1; then
    # Extract per-page data from JSON
    PAGE_COUNT=$(jq '.pages | length' "$JSON_TMPFILE")

    if [ "$PAGE_COUNT" -eq 0 ]; then
      # No pages - single text entry
      TEXT=$(jq -r '.text // ""' "$JSON_TMPFILE")
      jq -nc \
        --arg fn "data/$REL_PATH" \
        --arg doc "$REL_PATH" \
        --arg text "$TEXT" \
        '{file_name: $fn, document: $doc, page: 1, output_text: $text, output_json: {text: $text}}' \
        >> "$METADATA_FILE"
      TOTAL_ENTRIES=$((TOTAL_ENTRIES + 1))
      echo "  -> 1 text entry"
    else
      for i in $(seq 0 $((PAGE_COUNT - 1))); do
        PAGE_NUM=$(jq ".pages[$i].page" "$JSON_TMPFILE")
        PAGE_TEXT=$(jq -r ".pages[$i].text // \"\"" "$JSON_TMPFILE")

        jq -nc \
          --arg fn "data/$REL_PATH" \
          --arg doc "$REL_PATH" \
          --argjson page "$PAGE_NUM" \
          --arg text "$PAGE_TEXT" \
          --slurpfile json <(jq ".pages[$i]" "$JSON_TMPFILE") \
          '{file_name: $fn, document: $doc, page: $page, output_text: $text, output_json: $json[0]}' \
          >> "$METADATA_FILE"
        TOTAL_ENTRIES=$((TOTAL_ENTRIES + 1))
      done
      echo "  -> $PAGE_COUNT pages"
    fi
  else
    PARSE_ERROR=$(cat "$JSON_TMPFILE")
    echo "  ERROR: $PARSE_ERROR"
    jq -nc \
      --arg fn "data/$REL_PATH" \
      --arg doc "$REL_PATH" \
      --arg msg "$PARSE_ERROR" \
      '{file_name: $fn, document: $doc, page: 0, output_text: "", output_json: {error: true, message: $msg}}' \
      >> "$METADATA_FILE"
    TOTAL_ENTRIES=$((TOTAL_ENTRIES + 1))
  fi
  rm -f "$JSON_TMPFILE"
done < <(find "$SOURCE_DOCS_DIR" -type f -print0 | sort -z)

if [ "$TOTAL_ENTRIES" -eq 0 ]; then
  echo
  echo "ERROR: No dataset entries were generated. Check that source documents exist and are parseable."
  exit 1
fi

echo
echo "Dataset generation complete!"
echo "  Total entries: $TOTAL_ENTRIES"
echo "  Documents: $TOTAL_FILES"
echo "  Metadata: $METADATA_FILE"
echo "  Documents dir: $DOCUMENTS_DIR"
echo
echo "Use compare-dataset.sh to compare future output against this baseline."
