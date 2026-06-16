from src.cnnClassifier import logger

import os
import json
import zipfile
import gdown
import tensorflow as tf


# STAGE 1 — Data Ingestion
STAGE_NAME      = "Data Ingestion stage"
SOURCE_URL      = "https://drive.google.com/file/d/1z0mreUtRmR-P-magILsDR3T7M6IkGXtY/view?usp=sharing"
LOCAL_DATA_FILE = "artifacts/data_ingestion/data.zip"
UNZIP_DIR       = "artifacts/data_ingestion/"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    os.makedirs("artifacts/data_ingestion", exist_ok=True)

    file_id = SOURCE_URL.split("/")[-2]
    prefix  = "https://drive.google.com/uc?/export=download&id="
    logger.info(f"Downloading data from {SOURCE_URL}")
    gdown.download(prefix + file_id, LOCAL_DATA_FILE, quiet=False)
    logger.info(f"Downloaded to: {LOCAL_DATA_FILE}")

    logger.info(f"Extracting {LOCAL_DATA_FILE} ...")
    with zipfile.ZipFile(LOCAL_DATA_FILE, "r") as zip_ref:
        zip_ref.extractall(UNZIP_DIR)
    logger.info(f"Extracted to: {UNZIP_DIR}")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e

# STAGE 2 — Prepare Base Model
STAGE_NAME         = "Prepare base model"
ROOT_DIR           = "artifacts/prepare_base_model"
BASE_MODEL_PATH    = "artifacts/prepare_base_model/base_model.h5"
UPDATED_MODEL_PATH = "artifacts/prepare_base_model/base_model_updated.h5"

IMAGE_SIZE    = [224, 224, 3]
LEARNING_RATE = 0.01
INCLUDE_TOP   = False
WEIGHTS       = "imagenet"
CLASSES       = 2

try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    os.makedirs(ROOT_DIR, exist_ok=True)
    logger.info(f"Created directory: {ROOT_DIR}")

    logger.info("Loading VGG16 base model...")
    model = tf.keras.applications.VGG16(
        input_shape=IMAGE_SIZE,
        weights=WEIGHTS,
        include_top=INCLUDE_TOP
    )
    model.save(BASE_MODEL_PATH)
    logger.info(f"Base model saved to: {BASE_MODEL_PATH}")

    for layer in model.layers:
        layer.trainable = False
    logger.info("All layers frozen")

    flatten = tf.keras.layers.Flatten()(model.output)
    output  = tf.keras.layers.Dense(units=CLASSES, activation="softmax")(flatten)
    full_model = tf.keras.models.Model(inputs=model.input, outputs=output)

    full_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    full_model.summary()

    full_model.save(UPDATED_MODEL_PATH)
    logger.info(f"Updated model saved to: {UPDATED_MODEL_PATH}")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e

# STAGE 3 — Model Training
STAGE_NAME         = "Training"
TRAINED_MODEL_PATH = "artifacts/training/model.keras"
TRAINING_DATA      = "artifacts/data_ingestion/Chest-CT-Scan-data"

IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 16
EPOCHS        = 10
AUGMENTATION  = True
LEARNING_RATE = 0.01

try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    model = tf.keras.models.load_model(UPDATED_MODEL_PATH, compile=False)
    logger.info(f"Model loaded from: {UPDATED_MODEL_PATH}")

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    logger.info("Model compiled successfully")

    datagenerator_kwargs = dict(rescale=1./255, validation_split=0.20)
    dataflow_kwargs = dict(
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        interpolation="bilinear"
    )

    if AUGMENTATION:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            **datagenerator_kwargs
        )
        logger.info("Augmentation enabled")
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        logger.info("Augmentation disabled")

    train_generator = train_datagen.flow_from_directory(
        directory=TRAINING_DATA,
        subset="training",
        shuffle=True,
        **dataflow_kwargs
    )

    valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagenerator_kwargs
    ).flow_from_directory(
        directory=TRAINING_DATA,
        subset="validation",
        shuffle=False,
        **dataflow_kwargs
    )

    logger.info("Data generators created")

    steps_per_epoch  = train_generator.samples // BATCH_SIZE
    validation_steps = valid_generator.samples // BATCH_SIZE

    logger.info(f"Training for {EPOCHS} epoch(s)...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=valid_generator
    )

    os.makedirs(os.path.dirname(TRAINED_MODEL_PATH), exist_ok=True)
    model.save(TRAINED_MODEL_PATH)
    logger.info(f"Trained model saved to: {TRAINED_MODEL_PATH}")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e

# STAGE 4 — Evaluation
STAGE_NAME = "Evaluation stage"
SCORE_PATH = "scores.json"
BATCH_SIZE = 16

try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    model = tf.keras.models.load_model(TRAINED_MODEL_PATH)
    logger.info(f"Model loaded from: {TRAINED_MODEL_PATH}")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.20
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory=TRAINING_DATA,
        subset="validation",
        shuffle=False,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        interpolation="bilinear"
    )

    logger.info("Evaluating model...")
    loss, accuracy = model.evaluate(valid_generator)
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    scores = {"loss": round(loss, 4), "accuracy": round(accuracy, 4)}
    with open(SCORE_PATH, "w") as f:
        json.dump(scores, f, indent=4)
    logger.info(f"Scores saved to: {SCORE_PATH}")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e