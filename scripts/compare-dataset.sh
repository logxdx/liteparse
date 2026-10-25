#!/usr/bin/env bash
# Compares current liteparse output against a baseline dataset
#
# Usage:
#   ./scripts/compare-dataset.sh [dataset-dir]
#
# Exit codes:
#   0 - No changes detected
#   1 - Changes detected (requires approval)
#   2 - Error occurred
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_DIR="${1:-$REPO_ROOT/dataset}"
DOCUMENTS_DIR="$DATASET_DIR/data"
METADATA_FILE="$DATASET_DIR/metadata.jsonl"

LIT="$REPO_ROOT/target/release/liteparse"
if [ ! -x "$LIT" ]; then
  echo "ERROR: liteparse binary not found at $LIT. Run 'cargo build --release' first."
  exit 2
fi

echo "LiteParse Dataset Comparison"
echo "============================"
echo "Dataset: $DATASET_DIR"
echo "Documents: $DOCUMENTS_DIR"
echo

if [ ! -f "$METADATA_FILE" ]; then
  echo "ERROR: metadata.jsonl not found at $METADATA_FILE"
  exit 2
fi

ENTRY_COUNT=$(wc -l < "$METADATA_FILE" | tr -d ' ')
echo "Loaded $ENTRY_COUNT expected entries"
echo

if [ "$ENTRY_COUNT" -eq 0 ]; then
  echo "ERROR: metadata.jsonl is empty — dataset has no entries to compare against."
  echo "The dataset may need to be regenerated with: ./scripts/create-dataset.sh"
  exit 2
fi

DIFF_COUNT=0
ADDED_COUNT=0
REMOVED_COUNT=0
CHANGED_COUNT=0
DIFF_OUTPUT=""

# Collect unique documents from metadata
DOCUMENTS=$(jq -r '.document' "$METADATA_FILE" | sort -u)

while IFS= read -r document; do
  [ -z "$document" ] && continue
  FILE_PATH="$DOCUMENTS_DIR/$document"

  # Check if file exists
  if [ ! -f "$FILE_PATH" ]; then
    # All pages for this document are removed
    while IFS= read -r line; do
      PAGE=$(echo "$line" | jq -r '.page')
      DIFF_OUTPUT+="[REMOVED] $document (page $PAGE)"$'\n\n'
      REMOVED_COUNT=$((REMOVED_COUNT + 1))
      DIFF_COUNT=$((DIFF_COUNT + 1))
    done < <(jq -c "select(.document == \"$document\")" "$METADATA_FILE")
    continue
  fi

  echo "Checking: $document"

  # Parse current output
  CURRENT_JSON=""
  PARSE_ERROR=""
  if CURRENT_JSON=$("$LIT" parse --format json --no-ocr -q "$FILE_PATH" 2>&1); then
    # Compare each expected page
    while IFS= read -r line; do
      EXPECTED_PAGE=$(echo "$line" | jq -r '.page')
      EXPECTED_TEXT=$(echo "$line" | jq -r '.output_text')
      IS_PDF=false
      if [[ "$document" == *.pdf ]]; then
        IS_PDF=true
      fi

      # Get actual text for this page
      PAGE_EXISTS=$(echo "$CURRENT_JSON" | jq --argjson page "$EXPECTED_PAGE" \
        '[.pages[] | select(.page == $page)] | length' 2>/dev/null)
      ACTUAL_TEXT=$(echo "$CURRENT_JSON" | jq -r --argjson page "$EXPECTED_PAGE" \
        '.pages[] | select(.page == $page) | .text // ""' 2>/dev/null)

      if [ "$PAGE_EXISTS" = "0" ] && [ "$EXPECTED_PAGE" -ne 0 ]; then
        # Page not found in current output
        DIFF_OUTPUT+="[REMOVED] $document (page $EXPECTED_PAGE)"$'\n\n'
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
        DIFF_COUNT=$((DIFF_COUNT + 1))
        continue
      fi

      # Normalize for comparison
      if [ "$IS_PDF" = true ]; then
        EXPECTED_CMP=$(echo "$EXPECTED_TEXT" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        ACTUAL_CMP=$(echo "$ACTUAL_TEXT" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
      else
        # For non-PDF: collapse whitespace runs, trim lines, remove empty lines
        EXPECTED_CMP=$(echo "$EXPECTED_TEXT" | sed 's/[[:space:]]\{1,\}/ /g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed '/^$/d')
        ACTUAL_CMP=$(echo "$ACTUAL_TEXT" | sed 's/[[:space:]]\{1,\}/ /g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed '/^$/d')
      fi

      if [ "$EXPECTED_CMP" != "$ACTUAL_CMP" ]; then
        DIFF_OUTPUT+="[CHANGED] $document (page $EXPECTED_PAGE)"$'\n'
        # Show a unified-style diff
        DIFF_DETAIL=$(diff <(echo "$EXPECTED_TEXT") <(echo "$ACTUAL_TEXT") --unified=1 2>/dev/null || true)
        if [ -n "$DIFF_DETAIL" ]; then
          # Show first 50 lines of diff to avoid massive output
          DIFF_OUTPUT+="$(echo "$DIFF_DETAIL" | head -50)"$'\n'
          REMAINING=$(echo "$DIFF_DETAIL" | wc -l | tr -d ' ')
          if [ "$REMAINING" -gt 50 ]; then
            DIFF_OUTPUT+="... ($((REMAINING - 50)) more lines)"$'\n'
          fi
        fi
        DIFF_OUTPUT+=$'\n'
        CHANGED_COUNT=$((CHANGED_COUNT + 1))
        DIFF_COUNT=$((DIFF_COUNT + 1))
      fi
    done < <(jq -c "select(.document == \"$document\")" "$METADATA_FILE")

    # Check for new pages not in expected dataset
    EXPECTED_PAGES=$(jq -r "select(.document == \"$document\") | .page" "$METADATA_FILE" | sort -n)
    ACTUAL_PAGES=$(echo "$CURRENT_JSON" | jq -r '.pages[].page' | sort -n)

    while IFS= read -r page; do
      [ -z "$page" ] && continue
      if ! echo "$EXPECTED_PAGES" | grep -q "^${page}$"; then
        DIFF_OUTPUT+="[ADDED] $document (page $page)"$'\n\n'
        ADDED_COUNT=$((ADDED_COUNT + 1))
        DIFF_COUNT=$((DIFF_COUNT + 1))
      fi
    done <<< "$ACTUAL_PAGES"

  else
    PARSE_ERROR="$CURRENT_JSON"
    echo "  ERROR: $PARSE_ERROR"

    # Check if error was expected
    HAS_ERROR_ENTRY=$(jq -r "select(.document == \"$document\") | .output_json.error // false" "$METADATA_FILE" | head -1)
    if [ "$HAS_ERROR_ENTRY" != "true" ]; then
      DIFF_OUTPUT+="[CHANGED] $document (page 0)"$'\n'
      DIFF_OUTPUT+="  Expected: successful parse"$'\n'
      DIFF_OUTPUT+="  Actual: error: $PARSE_ERROR"$'\n\n'
      CHANGED_COUNT=$((CHANGED_COUNT + 1))
      DIFF_COUNT=$((DIFF_COUNT + 1))
    fi
  fi
done <<< "$DOCUMENTS"

echo
echo "Results"
echo "-------"

if [ "$DIFF_COUNT" -eq 0 ]; then
  echo "✓ No changes detected"
  exit 0
fi

echo "✗ $DIFF_COUNT change(s) detected:"
echo
echo "$DIFF_OUTPUT"

echo "---"
echo "SUMMARY:"
echo "  Added: $ADDED_COUNT"
echo "  Removed: $REMOVED_COUNT"
echo "  Changed: $CHANGED_COUNT"
echo
echo "This PR changes liteparse output and requires manual approval."

exit 1
