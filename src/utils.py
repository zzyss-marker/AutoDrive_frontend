from pathlib import Path

import numpy as np
import yaml


def as_intrinsics_matrix(intrinsics: list[float]):
    """
    Get matrix representation of intrinsics.
    intrinsics : [fx,fy,cx,cy]
    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def load_camera_cfg(cfg_path: str) -> dict:
    """
    Load camera configuration from YAML file.
    """
    assert Path(cfg_path).exists(), f"File not found: {cfg_path}"
    with open(cfg_path) as file:
        cfg = yaml.safe_load(file)
    return cfg
