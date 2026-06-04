#!/usr/bin/bash

if [[ $# -eq 0 ]]; then
    echo ""
    echo "Usage: $0 <L>"
    echo "L - maximum number of the latest runs"
    echo ""
    exit 1
fi

L="${1}"
counter=0
gh run list --repo SMLP-Systems/smlp --limit $L --json databaseId,displayTitle,headBranch \
  | jq -c '.[]' \
  | while read -r run; do
      id=$(echo "$run" | jq -r '.databaseId')
      title=$(echo "$run" | jq -r '.displayTitle')
      branch=$(echo "$run" | jq -r '.headBranch')
      counter=$((counter + 1))
      echo "=== $counter. Run $id [$branch]: $title ==="
      attempt_count=$(gh api "repos/SMLP-Systems/smlp/actions/runs/$id" --jq '.run_attempt')
      for i in $(seq 1 $attempt_count); do
          attempt_data=$(gh api "repos/SMLP-Systems/smlp/actions/runs/$id/attempts/$i" \
              --jq '{actor: .triggering_actor.login, created: .created_at, updated: .updated_at}')
          actor=$(echo "$attempt_data" | jq -r '.actor')
          created=$(date -d "$(echo "$attempt_data" | jq -r '.created')" '+%Y-%m-%d %H:%M:%S %Z')
          updated=$(date -d "$(echo "$attempt_data" | jq -r '.updated')" '+%Y-%m-%d %H:%M:%S %Z')
          artifacts=$(gh api "repos/SMLP-Systems/smlp/actions/runs/$id/artifacts" \
              --jq '.artifacts[].name')
          echo "  attempt $i [by: $actor] [started: $created] [finished: $updated]:"
          echo "$artifacts" | while read -r artifact; do
              echo "    $artifact"
          done
      done
    done
