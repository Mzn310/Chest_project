import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "artifacts/training/model.keras"
IMAGE_SIZE = (224, 224)
CLASSES    = ["adenocarcinoma", "normal"]  

TEST_CASES = [
    {"path": "tests/images/normal_1.jpg",         "expected": "normal"},
    {"path": "tests/images/normal_2.png",         "expected": "normal"},
    {"path": "tests/images/adenocarcinoma_1.png", "expected": "adenocarcinoma"},
    {"path": "tests/images/adenocarcinoma_2.png", "expected": "adenocarcinoma"},
]

def predict(image_path):
    img  = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    arr  = np.array(img) / 255.0
    arr  = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)
    return CLASSES[int(np.argmax(pred, axis=1)[0])], float(np.max(pred)) * 100

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded ✅\n")

passed = 0
failed = 0

for case in TEST_CASES:
    predicted, confidence = predict(case["path"])
    ok = predicted == case["expected"]

    if ok:
        passed += 1
        print(f" PASS | {case['path']}")
    else:
        failed += 1
        print(f" FAIL | {case['path']}")

    print(f"         Expected: {case['expected']} | Got: {predicted} | Confidence: {confidence:.1f}%\n")

print(f"── Results: {passed}/{len(TEST_CASES)} passed ──")

if failed > 0:
    print(f" {failed} test(s) failed — blocking deployment")
    raise SystemExit(1)

print("All tests passed ✅")