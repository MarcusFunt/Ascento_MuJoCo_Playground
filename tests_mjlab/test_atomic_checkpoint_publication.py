from pathlib import Path
from types import SimpleNamespace

from mjlab.rl import MjlabOnPolicyRunner

import ascento_mjlab.provenance_runner as provenance


def test_checkpoint_save_is_published_atomically(monkeypatch, tmp_path):
    writes = []
    uploaded = []

    def fake_base_save(self, path, infos=None):
        del self
        writes.append((Path(path), infos))
        Path(path).write_bytes(b"complete-checkpoint")

    monkeypatch.setattr(MjlabOnPolicyRunner, "save", fake_base_save)
    monkeypatch.setattr(provenance, "current_plant_contract", lambda: {"plant": 1})
    monkeypatch.setattr(provenance, "current_action_contract", lambda: {"action": 1})
    monkeypatch.setattr(
        provenance,
        "current_task_contract",
        lambda _cfg: {"task": 1},
    )

    runner = object.__new__(provenance.AscentoProvenanceRunner)
    runner.cfg = {"upload_model": True}
    runner.current_learning_iteration = 42
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=object()))
    runner.logger = SimpleNamespace(
        save_model=lambda path, iteration: uploaded.append((path, iteration))
    )
    runner._environment_progress = lambda: {"schema_version": 1}

    target = tmp_path / "model_42.pt"
    runner.save(str(target), infos={"custom": "value"})

    assert target.read_bytes() == b"complete-checkpoint"
    assert writes[0][0].parent == target.parent
    assert writes[0][0].name.endswith(".tmp")
    assert not writes[0][0].exists()
    assert writes[0][1]["custom"] == "value"
    assert uploaded == [(str(target.resolve()), 42)]
    assert runner.cfg["upload_model"] is True
