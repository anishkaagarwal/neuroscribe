"""
Smoke test for the Secure Feature Vault.

Run from the repo root:  python tests/test_lfv.py
Exits non-zero if any check fails, so CI catches regressions.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the project root importable when run as tests/test_lfv.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from secure_lfv import SecureFeatureVault  # noqa: E402


def test_lfv():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lfv = SecureFeatureVault(vault_dir=str(tmp / "vault"), user_password="test123")

        test_speaker = np.linspace(0.0, 1.0, 83)
        test_prosody = {
            "segments": [
                {"start": 0.0, "end": 2.5, "urgency": "High", "emotion": "frustration"},
                {"start": 2.5, "end": 5.0, "urgency": "Medium", "emotion": "concern"},
            ],
            "summary": {
                "high_urgency_count": 1,
                "emotions_detected": ["frustration", "concern"],
            },
        }
        test_intent = {
            "deepgram_intents": ["question", "complaint"],
            "urgent_action_items": ["Fix bug", "Schedule meeting"],
        }

        # 1. Store
        assert lfv.store_features(
            meeting_id="test_meeting_001",
            speaker_features=test_speaker,
            prosody_features=test_prosody,
            intent_features=test_intent,
            metadata={"filename": "test.mp4", "duration": 300},
        ), "store_features failed"

        # 2. Retrieve with integrity check, round-trip intact
        retrieved = lfv.retrieve_features("test_meeting_001", verify_integrity=True)
        assert retrieved, "retrieve_features returned nothing"
        assert np.allclose(retrieved["speaker"], test_speaker), "speaker features changed"
        assert retrieved["prosody"] == test_prosody, "prosody features changed"
        assert retrieved["intent"] == test_intent, "intent features changed"

        # 3. Listing and stats
        cached = [m["meeting_id"] for m in lfv.list_cached_meetings()]
        assert "test_meeting_001" in cached, f"meeting missing from cache list: {cached}"
        stats = lfv.get_statistics()
        assert stats["total_writes"] >= 1 and stats["total_reads"] >= 1, stats

        # 4. Delete
        assert lfv.delete_features("test_meeting_001"), "delete_features failed"
        assert lfv.retrieve_features("test_meeting_001") is None, "deleted meeting still readable"

        # 5. Key export
        key_path = tmp / "exported.key"
        assert lfv.export_key(str(key_path)), "export_key failed"
        assert key_path.exists(), "exported key file not written"

        lfv.clear_all()

    print("All Secure Feature Vault checks passed.")


if __name__ == "__main__":
    test_lfv()
