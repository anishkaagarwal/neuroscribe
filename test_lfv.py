"""
Test script for Secure Feature Vault
"""

import numpy as np
from secure_lfv import SecureFeatureVault

def test_lfv():
    print("🧪 Testing Secure Feature Vault\n")
    
    # Initialize vault
    lfv = SecureFeatureVault(vault_dir="test_vault", user_password="test123")
    
    # Create test features
    test_speaker = np.random.rand(83)
    test_prosody = {
        'segments': [
            {'start': 0.0, 'end': 2.5, 'urgency': 'High', 'emotion': 'frustration'},
            {'start': 2.5, 'end': 5.0, 'urgency': 'Medium', 'emotion': 'concern'}
        ],
        'summary': {
            'high_urgency_count': 1,
            'emotions_detected': ['frustration', 'concern']
        }
    }
    test_intent = {
        'deepgram_intents': ['question', 'complaint'],
        'urgent_action_items': ['Fix bug', 'Schedule meeting']
    }
    
    # Test 1: Store features
    print("1️⃣ Testing feature storage...")
    success = lfv.store_features(
        meeting_id="test_meeting_001",
        speaker_features=test_speaker,
        prosody_features=test_prosody,
        intent_features=test_intent,
        metadata={'filename': 'test.mp4', 'duration': 300}
    )
    print(f"   Storage: {'✅ PASS' if success else '❌ FAIL'}\n")
    
    # Test 2: Retrieve features
    print("2️⃣ Testing feature retrieval...")
    retrieved = lfv.retrieve_features("test_meeting_001", verify_integrity=True)
    
    if retrieved:
        print("   ✅ PASS - Retrieved features:")
        print(f"      Speaker shape: {retrieved['speaker'].shape}")
        print(f"      Prosody segments: {len(retrieved['prosody']['segments'])}")
        print(f"      Intent items: {len(retrieved['intent']['urgent_action_items'])}")
    else:
        print("   ❌ FAIL - Could not retrieve features")
    print()
    
    # Test 3: Integrity verification
    print("3️⃣ Testing integrity verification...")
    try:
        # Should pass
        retrieved = lfv.retrieve_features("test_meeting_001", verify_integrity=True)
        print("   ✅ PASS - Integrity check passed\n")
    except ValueError as e:
        print(f"   ❌ FAIL - Integrity check failed: {e}\n")
    
    # Test 4: List cached meetings
    print("4️⃣ Testing cache listing...")
    cached = lfv.list_cached_meetings()
    print(f"   Found {len(cached)} cached meetings:")
    for meeting in cached:
        print(f"      - {meeting['meeting_id']}")
    print()
    
    # Test 5: Statistics
    print("5️⃣ Testing statistics...")
    stats = lfv.get_statistics()
    print(f"   Total writes: {stats['total_writes']}")
    print(f"   Total reads: {stats['total_reads']}")
    print(f"   Hit rate: {stats['hit_rate']}")
    print()
    
    # Test 6: Deletion
    print("6️⃣ Testing feature deletion...")
    deleted = lfv.delete_features("test_meeting_001")
    print(f"   Deletion: {'✅ PASS' if deleted else '❌ FAIL'}\n")
    
    # Test 7: Key export
    print("7️⃣ Testing key export...")
    exported = lfv.export_key("test_exported_key.key")
    print(f"   Export: {'✅ PASS' if exported else '❌ FAIL'}\n")
    
    # Cleanup
    print("🧹 Cleaning up test vault...")
    lfv.clear_all()
    
    import shutil
    from pathlib import Path
    shutil.rmtree("test_vault", ignore_errors=True)
    Path("test_exported_key.key").unlink(missing_ok=True)
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_lfv()
