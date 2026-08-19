from __future__ import annotations

import json

import pytest

from plasticity_placement.p0d2hrr.runtime import _verify_crd_bank_audit


def _bank_audit() -> dict[str, object]:
    return {
        "schema_version": "p0d2hcrd-bank-audit-v1",
        "bank_sha256": "a" * 64,
        "all_checks_passed": True,
        "records": [
            {
                "probe_id": "probe-1",
                "ordered_candidates": ("slot_a", "slot_b"),
            }
        ],
    }


def test_crd_bank_audit_accepts_json_tuple_list_normalization() -> None:
    regenerated = _bank_audit()
    stored = json.loads(json.dumps(regenerated))

    assert regenerated != stored
    _verify_crd_bank_audit(regenerated, stored)


def test_crd_bank_audit_still_rejects_real_content_changes() -> None:
    regenerated = _bank_audit()
    stored = json.loads(json.dumps(regenerated))
    stored["records"][0]["ordered_candidates"][0] = "slot_changed"

    with pytest.raises(ValueError, match="CRD bank changed"):
        _verify_crd_bank_audit(regenerated, stored)
