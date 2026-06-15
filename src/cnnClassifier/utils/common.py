import os
from box.expceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)
            if not content:
                raise ValueError("yaml file is empty")
            logger.info(f"yaml loaded: {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(paths: list, verbose=True):
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created dir: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json saved: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json loaded: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    joblib.dump(value=data, filename=path)
    logger.info(f"bin saved: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logger.info(f"bin loaded: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    return f"~ {round(os.path.getsize(path)/1024)} KB"

def decodeImage(imgstring, fileName):
    with open(fileName, 'wb') as f:
        f.write(base64.b64decode(imgstring))

def encodeImageIntoBase64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read())