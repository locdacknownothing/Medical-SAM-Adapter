#!/usr/bin/env bash
#
# validate-commit-msg.sh
# Validates a commit message string or file according to Conventional Commits.
# Usage:
#   ./validate-commit-msg.sh "feat(scope): short description"
#   ./validate-commit-msg.sh path/to/COMMIT_EDITMSG
#

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <commit-message-string | path-to-commit-msg-file>" >&2
    exit 1
fi

INPUT="$1"

if [ -f "$INPUT" ]; then
    # Read first non-comment line as header
    HEADER=$(grep -v '^#' "$INPUT" | sed '/^\s*$/d' | head -n 1 || true)
    CONTENT=$(grep -v '^#' "$INPUT" || true)
else
    HEADER=$(echo "$INPUT" | head -n 1)
    CONTENT="$INPUT"
fi

if [ -z "$HEADER" ]; then
    echo "ERROR: Commit message header is empty." >&2
    exit 1
fi

# Check maximum header length (72 characters max)
HEADER_LEN=${#HEADER}
if [ "$HEADER_LEN" -gt 72 ]; then
    echo "ERROR: Header length is $HEADER_LEN characters (must be <= 72 characters)." >&2
    echo "Header: '$HEADER'" >&2
    exit 1
fi

# Check trailing period in header
if [[ "$HEADER" =~ \.$ ]]; then
    echo "ERROR: Header must not end with a period." >&2
    echo "Header: '$HEADER'" >&2
    exit 1
fi

# Regex for Conventional Commits header:
# type(scope)!: description OR type!: description OR type(scope): description OR type: description
REGEX="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-zA-Z0-9_\.\/-]+\))?(!)?: [a-zA-Z0-9].*$"

if ! [[ "$HEADER" =~ $REGEX ]]; then
    echo "ERROR: Header does not match Conventional Commits format." >&2
    echo "Expected format: <type>(<scope>): <description> OR <type>: <description>" >&2
    echo "Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert" >&2
    echo "Received: '$HEADER'" >&2
    exit 1
fi

# Check if second line is non-empty when there are multiple lines
LINE_COUNT=$(echo "$CONTENT" | wc -l)
if [ "$LINE_COUNT" -gt 1 ]; then
    SECOND_LINE=$(echo "$CONTENT" | sed -n '2p')
    if [ -n "$SECOND_LINE" ]; then
        echo "ERROR: Second line must be blank to separate header and body." >&2
        exit 1
    fi
fi

echo "SUCCESS: Commit message format is valid."
exit 0
