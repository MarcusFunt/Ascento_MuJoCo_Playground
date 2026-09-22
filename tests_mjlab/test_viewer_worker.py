import threading
from types import SimpleNamespace

import pytest
import torch
from mjlab.viewer.base import ViewerAction

import ascento_mjlab.viewer.worker as worker


class _FakeAlg:
    def __init__(self):
        self.actor = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.actor.weight.fill_(1.0)

    def get_policy(self):
        return self.actor

    def eval_mode(self):
        self.actor.eval()


class _FakeRunner:
    def __init__(self, *, fail_after_mutation=False):
        self.alg = _FakeAlg()
        self.fail_after_mutation = fail_after_mutation
        self.load_called = False

    def load(self, *_args, **_kwargs):
        self.load_called = True
        with torch.no_grad():
            self.alg.actor.weight.fill_(9.0)
        if self.fail_after_mutation:
            raise RuntimeError("load failed after mutation")
        return {"contract": "ok"}


def test_contract_rejection_happens_before_live_actor_mutation(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model_200.pt"
    torch.save({"infos": {"contract": "bad"}}, checkpoint)
    runner = _FakeRunner()
    monkeypatch.setattr(
        worker,
        "require_current_checkpoint_contracts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("contract mismatch")),
    )

    with pytest.raises(ValueError, match="contract mismatch"):
        worker._load_actor_transactionally(
            runner,
            checkpoint,
            object(),
            device="cpu",
        )

    assert runner.load_called is False
    assert runner.alg.actor.weight.item() == pytest.approx(1.0)


def test_failed_actor_load_restores_previous_policy_state(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model_200.pt"
    torch.save({"infos": {"contract": "ok"}}, checkpoint)
    runner = _FakeRunner(fail_after_mutation=True)
    monkeypatch.setattr(
        worker,
        "require_current_checkpoint_contracts",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="load failed after mutation"):
        worker._load_actor_transactionally(
            runner,
            checkpoint,
            object(),
            device="cpu",
        )

    assert runner.alg.actor.weight.item() == pytest.approx(1.0)


def test_failed_hot_swap_keeps_viewer_policy_and_current_checkpoint():
    viewer = worker._FollowViserPlayViewer.__new__(worker._FollowViserPlayViewer)
    original_policy = object()
    viewer.policy = original_policy
    viewer._last_error = None
    viewer._follow_rejected_checkpoint = None
    viewer._ckpt_user_event = threading.Event()
    viewer._ckpt_user_event.set()
    viewer._ckpt_dropdown = SimpleNamespace(
        options=["model_100.pt", "model_200.pt"],
        value="model_100.pt",
    )

    class Manager:
        current_name = "model_100.pt"

        @staticmethod
        def fetch_available():
            return [("model_100.pt", ""), ("model_200.pt", "")]

        @staticmethod
        def load_checkpoint(_name):
            raise ValueError("incompatible checkpoint")

    viewer._ckpt_mgr = Manager()

    handled = viewer._handle_custom_action(ViewerAction.FETCH_CHECKPOINT, "latest")

    assert handled is True
    assert viewer.policy is original_policy
    assert viewer._ckpt_mgr.current_name == "model_100.pt"
    assert viewer._follow_rejected_checkpoint == "model_200.pt"
    assert "incompatible checkpoint" in viewer._last_error
