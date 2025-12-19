import os
from pathlib import Path       ## Path lib will handle path for different os(windows/linus)
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]:%(message)s")         ## Decides the logging style -> [2025-11-29 11:43:16,649]:logging starts 


list_of_files=[
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/logging/__init__.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/pipeline/__init__.py",
    "src/entity/__init__.py",
    "src/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "requirements.txt",
    "setup.py",
    "reasearch/trials.ipynb"

]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)        ## Splits the folder/files to folder,file_name
    
    if filedir!="" :
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if(not  os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as fobj:
            
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} already exists")


