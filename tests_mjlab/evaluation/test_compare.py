import json

import pytest

from ascento_mjlab.evaluation.compare import ensure_compatible_plants
from ascento_mjlab.plant_contract import current_plant_contract


def _manifest(directory, contract):
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps({"plant_contract": contract, "checkpoint_plant_contract": contract}),
        encoding="utf-8",
    )


def test_comparison_rejects_legacy_or_mixed_plant_provenance(tmp_path):
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    _manifest(legacy, None)
    _manifest(current, current_plant_contract())

    with pytest.raises(ValueError, match="legacy"):
        ensure_compatible_plants(legacy, current)


def test_comparison_accepts_exact_matching_plant_contracts(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    contract = current_plant_contract()
    _manifest(base, contract)
    _manifest(candidate, contract)

    ensure_compatible_plants(base, candidate)
