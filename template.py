import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s'
)

project_name = "cnnClassifier"

list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]


for file in list_of_files:
    # Converts string into a Path object
    file=Path(file)
    
    file_dir,file_name=os.path.split(file)


    if file_dir!="":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")
    
    # It exists but is empty
    if (not os.path.exists(file) or (os.path.getsize(file)==0)):
        with open(file,'w') as f:
            pass
        logging.info(f"Creating empty file: {file}")
    
    else:
        logging.info(f"File: {file} already exists and is not empty. Skipping creation.")