import os    
from box.exceptions import BoxValueError    # Raise  box exceptions error
import yaml                                 # To read yaml file using safe_load
from src.logger import logger
from ensure import ensure_annotations       # To raise error for data types mismatch
from box import ConfigBox                   #  Convert  student["id]"  ---> student.id
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path)->ConfigBox:
    """reads yaml file -> used to store settins,path,parametsers

    Args:
        path_to_yaml (str): Path like input
    Raises:
        pass
    Returns:
        ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content=yaml.safe_load(yaml_file)
            logger.info(f"yaml file:{path_to_yaml} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise KeyError("Check if a key exists before accessing")
    
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

    