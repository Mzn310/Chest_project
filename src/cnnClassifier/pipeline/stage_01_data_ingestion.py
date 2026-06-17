import os
import zipfile
import gdown
from cnnClassifier.logger import logger

STAGE_NAME      = "Data Ingestion stage"
SOURCE_URL      = "https://drive.google.com/file/d/1z0mreUtRmR-P-magILsDR3T7M6IkGXtY/view?usp=sharing"
LOCAL_DATA_FILE = "artifacts/data_ingestion/data.zip"
UNZIP_DIR       = "artifacts/data_ingestion/"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    os.makedirs("artifacts/data_ingestion", exist_ok=True)

    file_id = SOURCE_URL.split("/")[-2]
    logger.info(f"Downloading data from {SOURCE_URL}")
    gdown.download(id=file_id, output=LOCAL_DATA_FILE, quiet=False)
    logger.info(f"Downloaded to: {LOCAL_DATA_FILE}")

    logger.info(f"Extracting {LOCAL_DATA_FILE} ...")
    with zipfile.ZipFile(LOCAL_DATA_FILE, "r") as zip_ref:
        zip_ref.extractall(UNZIP_DIR)
    logger.info(f"Extracted to: {UNZIP_DIR}")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e