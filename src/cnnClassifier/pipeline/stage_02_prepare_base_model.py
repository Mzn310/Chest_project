import os
import logging
import tensorflow as tf
from cnnClassifier.logger import logger


STAGE_NAME         = "Prepare base model"
ROOT_DIR           = "artifacts/prepare_base_model"
BASE_MODEL_PATH    = "artifacts/prepare_base_model/base_model.keras"
UPDATED_MODEL_PATH = "artifacts/prepare_base_model/base_model_updated.keras"

IMAGE_SIZE    = [224, 224, 3]
LEARNING_RATE = 0.01
INCLUDE_TOP   = False
WEIGHTS       = "imagenet"
CLASSES       = 2


try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    # Step 1: Create directory
    os.makedirs(ROOT_DIR, exist_ok=True)
    logger.info(f"Created directory: {ROOT_DIR}")

    # Step 2: Load base VGG16 model
    logger.info("Loading VGG16 base model...")
    model = tf.keras.applications.VGG16(
        input_shape=IMAGE_SIZE,
        weights=WEIGHTS,
        include_top=INCLUDE_TOP
    )
    model.save(BASE_MODEL_PATH)
    logger.info(f"Base model saved to: {BASE_MODEL_PATH}")

    # Step 3: Freeze all layers
    for layer in model.layers:
        layer.trainable = False
    logger.info("All layers frozen")

    # Step 4: Add custom head
    flatten = tf.keras.layers.Flatten()(model.output)
    output  = tf.keras.layers.Dense(units=CLASSES, activation="softmax")(flatten)

    full_model = tf.keras.models.Model(inputs=model.input, outputs=output)

    # Step 5: Compile
    full_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    full_model.summary()

    # Step 6: Save updated model
    full_model.save(UPDATED_MODEL_PATH)
    logger.info(f"Updated model saved to: {UPDATED_MODEL_PATH}")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e
