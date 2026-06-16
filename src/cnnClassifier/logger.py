import os
import logging

# create logs folder once
os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/running_logs.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("cnnClassifierLogger")