import base64
import os

# ── Test 1: base64 decode works ──
def test_decode_image():
    with open("tests/images/normal_1.jpg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    decoded = base64.b64decode(encoded)
    assert len(decoded) > 0, "Decoded image should not be empty"
    print("✅ test_decode_image passed")

# ── Test 2: artifacts folders exist after training ──
def test_artifacts_exist():
    assert os.path.exists("artifacts/data_ingestion"),       "data_ingestion folder missing"
    assert os.path.exists("artifacts/prepare_base_model"),   "prepare_base_model folder missing"
    assert os.path.exists("artifacts/training"),             "training folder missing"
    print("✅ test_artifacts_exist passed")

# ── Test 3: model file exists ──
def test_model_file_exists():
    assert os.path.exists("artifacts/training/model.keras"), "model.keras not found"
    print("test_model_file_exists passed")

# ── Test 4: scores.json exists and has valid keys ──
def test_scores_json():
    import json
    assert os.path.exists("scores.json"), "scores.json not found"
    with open("scores.json") as f:
        scores = json.load(f)
    assert "loss" in scores,     "loss key missing in scores.json"
    assert "accuracy" in scores, "accuracy key missing in scores.json"
    assert 0 <= scores["accuracy"] <= 1, "accuracy should be between 0 and 1"
    print(" test_scores_json passed")

if __name__ == "__main__":
    failed = 0
    for name, fn in [
        ("test_decode_image",     test_decode_image),
        ("test_artifacts_exist",  test_artifacts_exist),
        ("test_model_file_exists",test_model_file_exists),
        ("test_scores_json",      test_scores_json),
    ]:
        try:
            fn()
        except Exception as e:
            print(f" {name} FAILED: {e}")
            failed += 1

    print(f"\n── Results: {4 - failed}/4 passed ──")
    if failed > 0:
        raise SystemExit(1)
    print("All unit tests passed ")