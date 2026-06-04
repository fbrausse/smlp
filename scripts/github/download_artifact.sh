#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo ""
    echo "Usage: $0 <artifact-name> [run-id] [outdir]"
    echo ""
    exit 1
fi

REPO="SMLP-Systems/smlp"
ARTIFACT_NAME="$1"
RUN_ID="${2:-}"
OUTDIR="${3:-.}"

if [[ -z "$RUN_ID" ]]; then
    echo "INFO: Fetching latest run ID"
    RUN_ID=$(gh run list --repo "$REPO" --limit 1 --json databaseId --jq '.[0].databaseId')
fi

echo "INFO: Run ID: $RUN_ID"

ARTIFACT_ID=$(gh api "repos/$REPO/actions/runs/$RUN_ID/artifacts" \
    | jq -r --arg name "$ARTIFACT_NAME" 'first(.artifacts[] | select(.name == $name) | .id)')

if [[ -z "$ARTIFACT_ID" ]]; then
    echo "ERROR: Artifact '$ARTIFACT_NAME' not found in run $RUN_ID"
    exit 1
fi

echo "INFO: Artifact ID: $ARTIFACT_ID"

OUTFILE="$OUTDIR/$ARTIFACT_NAME"
echo "INFO: Downloading to $OUTFILE"
gh api "repos/$REPO/actions/artifacts/$ARTIFACT_ID/zip" > "$OUTFILE"

echo "INFO: Done — $OUTFILE"
