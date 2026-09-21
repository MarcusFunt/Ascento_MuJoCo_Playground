import hashlib
import json
import subprocess

import dashboard.provenance as provenance
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


def test_state_uses_verified_image_provenance_without_git_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCENTO_REPOSITORY_COMMIT", "image-commit")
    monkeypatch.setenv("ASCENTO_REPOSITORY_BRANCH", "main")
    monkeypatch.setenv("ASCENTO_REPOSITORY_DIRTY", "0")

    state = working_tree_state(tmp_path)

    assert state.commit == "image-commit"
    assert state.branch == "main"
    assert not state.is_dirty


@pytest.mark.parametrize(
    ("commit", "branch"),
    [
        (None, "main"),
        ("", "main"),
        ("unknown", "main"),
        ("image-commit", None),
        ("image-commit", ""),
        ("image-commit", "unknown"),
    ],
)
def test_state_rejects_image_provenance_without_known_commit_and_branch(
    tmp_path, monkeypatch, commit, branch
):
    if commit is None:
        monkeypatch.delenv("ASCENTO_REPOSITORY_COMMIT", raising=False)
    else:
        monkeypatch.setenv("ASCENTO_REPOSITORY_COMMIT", commit)
    if branch is None:
        monkeypatch.delenv("ASCENTO_REPOSITORY_BRANCH", raising=False)
    else:
        monkeypatch.setenv("ASCENTO_REPOSITORY_BRANCH", branch)
    monkeypatch.setenv("ASCENTO_REPOSITORY_DIRTY", "0")

    with pytest.raises(ValueError, match="verified clean image provenance"):
        working_tree_state(tmp_path)


@pytest.mark.parametrize("dirty_status", [None, "1", "unknown"])
def test_state_rejects_unverified_image_provenance(tmp_path, monkeypatch, dirty_status):
    monkeypatch.setenv("ASCENTO_REPOSITORY_COMMIT", "image-commit")
    monkeypatch.setenv("ASCENTO_REPOSITORY_BRANCH", "main")
    if dirty_status is None:
        monkeypatch.delenv("ASCENTO_REPOSITORY_DIRTY", raising=False)
    else:
        monkeypatch.setenv("ASCENTO_REPOSITORY_DIRTY", dirty_status)

    with pytest.raises(ValueError, match="verified clean image provenance"):
        working_tree_state(tmp_path)


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


def test_bundle_accepts_mixed_staged_and_unstaged_changes(initialized_repo, tmp_path):
    tracked = initialized_repo / "tracked.py"
    subprocess.run(["git", "add", "tracked.py"], cwd=initialized_repo, check=True)
    tracked.write_text("after = 3\n", encoding="utf-8")

    bundle = write_dirty_source_bundle(initialized_repo, tmp_path / "bundle")

    assert "tracked.py" in bundle["tracked_paths"]
    assert bundle["tracked_patch"]["sha256"]


def test_failed_bundle_capture_does_not_leave_partial_directory(initialized_repo, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("outside = True\n", encoding="utf-8")
    (initialized_repo / "escape.py").symlink_to(outside)
    destination = tmp_path / "partial-bundle"

    with pytest.raises(ValueError, match="escapes repository root"):
        write_dirty_source_bundle(initialized_repo, destination)

    assert not destination.exists()


def test_bundle_rejects_patch_missing_a_tracked_path(initialized_repo, tmp_path, monkeypatch):
    original_git = provenance._git

    def incomplete_diff(repo_root, *args, env=None):
        if args == ("diff", "--binary", "--no-renames", "HEAD"):
            return ""
        return original_git(repo_root, *args, env=env)

    monkeypatch.setattr(provenance, "_git", incomplete_diff)
    destination = tmp_path / "incomplete-bundle"

    with pytest.raises(ValueError, match="does not cover the tracked source snapshot"):
        write_dirty_source_bundle(initialized_repo, destination)

    assert not destination.exists()


def test_bundle_rejects_source_changes_during_capture(initialized_repo, tmp_path, monkeypatch):
    original_untracked_manifest = provenance._untracked_manifest
    tracked = initialized_repo / "tracked.py"
    calls = 0

    def mutate_after_snapshot(repo_root, paths):
        nonlocal calls
        records = original_untracked_manifest(repo_root, paths)
        calls += 1
        if calls == 1:
            tracked.write_text("changed during capture = True\n", encoding="utf-8")
        return records

    monkeypatch.setattr(provenance, "_untracked_manifest", mutate_after_snapshot)
    destination = tmp_path / "unstable-bundle"

    with pytest.raises(ValueError, match="source tree changed during provenance capture"):
        write_dirty_source_bundle(initialized_repo, destination)

    assert not destination.exists()
