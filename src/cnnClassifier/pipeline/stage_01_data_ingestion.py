import os
import zipfile
import gdown
import logging

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

STAGE_NAME     = "Data Ingestion stage"
SOURCE_URL     = "https://drive.google.com/file/d/1z0mreUtRmR-P-magILsDR3T7M6IkGXtY/view?usp=sharing"
LOCAL_DATA_FILE = "artifacts/data_ingestion/data.zip"
UNZIP_DIR      = "artifacts/data_ingestion/"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    # Step 1
    os.makedirs("artifacts/data_ingestion", exist_ok=True)

    # Step 2
    file_id = SOURCE_URL.split("/")[-2]
    prefix  = "https://drive.google.com/uc?/export=download&id="
    logger.info(f"Downloading data from {SOURCE_URL}")
    gdown.download(prefix + file_id, LOCAL_DATA_FILE, quiet=False)
    logger.info(f"Downloaded to: {LOCAL_DATA_FILE}")

    # Step 3:
    logger.info(f"Extracting {LOCAL_DATA_FILE} ...")
    with zipfile.ZipFile(LOCAL_DATA_FILE, "r") as zip_ref:
        zip_ref.extractall(UNZIP_DIR)
    logger.info(f"Extracted to: {UNZIP_DIR}")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e