from typing import Any, Dict

import yaml

from app.utils.path import (
    DATA_CONFIG_PATH,
    FEATURE_CONFIG_PATH,
    MODEL_CONFIG_PATH,
    TRAIN_CONFIG_PATH,
)


def load_config(path: str) -> Dict[str, Any]:
    """Configuration loader.

    Description:
        Load configuration yaml file into python dictionary.

    Args:
        path (str): Configuration path.

    Returns:
        (Dict[str, Any]): Dictionary of configuration.
    """
    config = {}
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_all_configs():
    """
    Load various configuration files required for data processing and model training.
    Depending on the data_type, different training configuration paths are used.
    """

    configs = {
        "data": load_config(path=DATA_CONFIG_PATH),
        "model": load_config(path=MODEL_CONFIG_PATH),
        "train": load_config(path=TRAIN_CONFIG_PATH),
        "feature": load_config(path=FEATURE_CONFIG_PATH),
    }

    return configs


def load_yaml(file_path: str) -> dict:
    """
    YAML 파일을 로드하여 파이썬 딕셔너리로 반환하는 함수.

    Args:
        file_path (str): 로드할 YAML 파일의 경로.

    Returns:
        dict: YAML 파일의 내용을 담은 파이썬 딕셔너리 객체.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dump_yaml(file_path: str) -> None:
    """
    파이썬 딕셔너리를 YAML 형식으로 지정된 파일에 덤프하는 함수.

    Args:
        file_path (str): 덤프할 YAML 파일의 경로.

    Returns:
        None
    """
    return yaml.dump(file_path, sort_keys=False, allow_unicode=True)
