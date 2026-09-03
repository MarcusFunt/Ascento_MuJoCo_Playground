"""Backfill repository provenance for legacy runs that predate launcher metadata."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dashboard.health import discover_dashboard_runs
from dashboard.versioning import run_repository_provenance


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
