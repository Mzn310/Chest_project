import os
import logging
import tensorflow as tf

# ─── Logger Setup ─────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
    handlers=[
        logging.FileHandler("logs/running_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cnnClassifierLogger")

# ─── Config ───────────────────────────────────────────
STAGE_NAME         = "Training"
BASE_MODEL_PATH    = "artifacts/prepare_base_model/base_model_updated.h5"
TRAINED_MODEL_PATH = "artifacts/training/model.keras"
TRAINING_DATA      = "artifacts/data_ingestion/Chest-CT-Scan-data"

IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 16
EPOCHS        = 10
AUGMENTATION  = True
LEARNING_RATE = 0.01

# ─── Pipeline ─────────────────────────────────────────
try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    # Step 1: Load model
    model = tf.keras.models.load_model(BASE_MODEL_PATH, compile=False)
    logger.info(f"Model loaded from: {BASE_MODEL_PATH}")

    # ✅ Step 2: Compile (fixes the error)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    logger.info("Model compiled successfully")

    # Step 3: Data generators
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

    # Step 4: Train
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

    # Step 5: Save
    os.makedirs(os.path.dirname(TRAINED_MODEL_PATH), exist_ok=True)
    model.save(TRAINED_MODEL_PATH)
    logger.info(f"Trained model saved to: {TRAINED_MODEL_PATH}")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e