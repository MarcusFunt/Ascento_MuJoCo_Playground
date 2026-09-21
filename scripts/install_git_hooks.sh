#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
source_hook="$repo_root/scripts/git-hooks/pre-commit"
target_hook=$(git rev-parse --git-path hooks/pre-commit)

if [[ -e "$target_hook" || -L "$target_hook" ]]; then
  if cmp -s "$source_hook" "$target_hook"; then
    chmod 0755 "$target_hook"
    printf 'Pre-commit hook already installed at %s\n' "$target_hook"
    exit 0
  fi
  printf 'A different pre-commit hook already exists at %s; preserving the existing hook.\n' \
    "$target_hook" >&2
  exit 1
fi

install -m 0755 "$source_hook" "$target_hook"
printf 'Installed pre-commit hook at %s\n' "$target_hook"
