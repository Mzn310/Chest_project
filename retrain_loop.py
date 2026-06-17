import json
import subprocess
import sys

MAX_ATTEMPTS = 3
THRESHOLD = 0.5

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"Attempt {attempt}/{MAX_ATTEMPTS}: running pipeline...")
    
    # Force rerun of training + evaluation, ignoring cache
    result = subprocess.run(
        ["dvc", "repro", "--force", "model_trainer", "model_evaluation"],
        capture_output=False
    )
    if result.returncode != 0:
        print("Pipeline run failed.")
        sys.exit(1)

    with open("scores.json") as f:
        scores = json.load(f)

    accuracy = scores.get("accuracy", 0)
    print(f"Accuracy: {accuracy}")

    if accuracy >= THRESHOLD:
        print("Accuracy meets threshold. Done.")
        sys.exit(0)

    print(f"Accuracy below {THRESHOLD}, retrying training...")

print("Max attempts reached without meeting threshold.")
sys.exit(1)