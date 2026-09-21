#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
source_hook="$repo_root/scripts/git-hooks/pre-commit"
target_hook="$repo_root/.git/hooks/pre-commit"

install -m 0755 "$source_hook" "$target_hook"
printf 'Installed pre-commit hook at %s\\n' "$target_hook"
