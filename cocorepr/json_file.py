# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_json_file.ipynb (unless otherwise specified).

__all__ = ['logger', 'load_json_file', 'dump_json_file']

# Cell

import logging
from dataclasses import dataclass
from typing import *
from pathlib import Path
import json

from .utils import sort_dict
from .coco import *

logger = logging.getLogger()

# Cell

def load_json_file(annotations_json: Union[str, Path], *, kind: str = "object_detection") -> CocoDataset:
    from_dict_function = get_dataset_class(kind).from_dict

    annotations_json = Path(annotations_json)
    logger.info(f"Loading json_file from: {annotations_json}")
    ext = annotations_json.suffix
    if ext != '.json':
        raise ValueError(f'Expect .json file as input, got: {annotations_json}')

    D = json.loads(annotations_json.read_text())
    coco = from_dict_function(D)
    logger.info(f"Loaded from json_file: {coco.to_full_str()}")
    return coco

# Cell

def dump_json_file(
    coco: CocoDataset,
    annotations_json: Union[str, Path],
    *,
    kind: str = "object_detection",
    skip_nulls: bool = False,
    overwrite: bool = False,
    indent: Optional[int] = 4,
) -> None:
    dataset_class = get_dataset_class(kind)
    if skip_nulls:
        to_dict_function = dataset_class.to_dict_skip_nulls
    else:
        to_dict_function = dataset_class.to_dict

    annotations_json = Path(annotations_json)
    if annotations_json.is_file() and not overwrite:
        raise ValueError(f"Destination json_file already exists: {annotations_json}")
    raw = sort_dict(to_dict_function(coco))
    logger.info(f"Writing dataset {coco.to_full_str()} to json-file: {annotations_json}")
    annotations_json.parent.mkdir(parents=True, exist_ok=True)
    annotations_json.write_text(json.dumps(raw, indent=indent))