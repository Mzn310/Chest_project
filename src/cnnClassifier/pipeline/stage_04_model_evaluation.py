import os
import json
import tensorflow as tf
from src.cnnClassifier.logger import logger


# ─── Config ───────────────────────────────────────────
STAGE_NAME         = "Evaluation stage"
TRAINED_MODEL_PATH = "artifacts/training/model.keras"
TRAINING_DATA      = "artifacts/data_ingestion/Chest-CT-Scan-data"
SCORE_PATH         = "scores.json"

IMAGE_SIZE  = [224, 224, 3]
BATCH_SIZE  = 16

# ─── Pipeline ─────────────────────────────────────────
try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    # Step 1: Load trained model
    logger.info(f"Loading model from: {TRAINED_MODEL_PATH}")
    model = tf.keras.models.load_model(TRAINED_MODEL_PATH)
    logger.info("Model loaded successfully")

    # Step 2: Validation generator
    logger.info("Setting up validation data generator...")
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.20
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory=TRAINING_DATA,
        subset="validation",
        shuffle=False,
        target_size=IMAGE_SIZE[:-1],
        batch_size=BATCH_SIZE,
        interpolation="bilinear"
    )

    # Step 3: Evaluate
    logger.info("Evaluating model...")
    loss, accuracy = model.evaluate(valid_generator)
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Step 4: Save scores
    scores = {"loss": round(loss, 4), "accuracy": round(accuracy, 4)}
    with open(SCORE_PATH, "w") as f:
        json.dump(scores, f, indent=4)
    logger.info(f"Scores saved to: {SCORE_PATH}")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e