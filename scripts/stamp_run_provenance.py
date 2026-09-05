"""Backfill repository provenance for legacy runs that predate launcher metadata."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# This script is executed directly by scripts/maintain.sh. In that mode Python
# puts ``scripts/`` (not the repository root) on sys.path, while ``dashboard``
# is a top-level repository package rather than part of the installed src
# package. Add the checkout root explicitly so a fresh/updated installation can
# import the dashboard helpers without relying on the caller's working directory
# or PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.health import discover_dashboard_runs  # noqa: E402
from dashboard.versioning import run_repository_provenance  # noqa: E402


def stamp_missing_runs(root: Path, commit: str, branch: str | None = None) -> int:
    root = root.expanduser().resolve()
    if not root.exists() or not commit:
        return 0

    stamped = 0
    for ref in discover_dashboard_runs(root):
        existing = run_repository_provenance(ref.path, root)
        if existing.get("commit"):
            continue
        payload = {
            "schema_version": 1,
            "commit": commit,
            "branch": branch,
            "inferred": True,
            "inference": "checkout present immediately before maintenance update",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        destination = ref.path / "repository_provenance.json"
        tmp = destination.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(destination)
        stamped += 1
    return stamped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--branch")
    args = parser.parse_args()
    count = stamp_missing_runs(args.root, args.commit, args.branch)
    print(f"Stamped repository provenance for {count} legacy run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
