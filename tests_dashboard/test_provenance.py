import hashlib
import json
import subprocess

import pytest
from dashboard.provenance import working_tree_state, write_dirty_source_bundle


@pytest.fixture
def initialized_repo(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    (repo / "tracked.py").write_text("before = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "fixture"], check=True)
    (repo / "tracked.py").write_text("after = 2\n", encoding="utf-8")
    (repo / "new.py").write_text("new = 3\n", encoding="utf-8")
    return repo


def test_state_reports_tracked_and_untracked_paths(initialized_repo):
    state = working_tree_state(initialized_repo)

    assert state.tracked_paths == ("tracked.py",)
    assert state.untracked_paths == ("new.py",)
    assert state.commit
    assert state.branch


def test_bundle_has_valid_patch_and_hashed_untracked_file(initialized_repo, tmp_path):
    destination = tmp_path / "bundle"

    bundle = write_dirty_source_bundle(initialized_repo, destination)

    patch = destination / "tracked.patch"
    assert patch.is_file()
    subprocess.run(
        ["git", "apply", "--check", "--cached", str(patch)],
        cwd=initialized_repo,
        check=True,
    )
    assert (destination / "untracked.tar").is_file()
    assert bundle["untracked"]["new.py"]["sha256"] == hashlib.sha256(
        b"new = 3\n"
    ).hexdigest()
    assert json.loads((destination / "manifest.json").read_text(encoding="utf-8")) == bundle
