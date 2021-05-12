# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_json_file.ipynb (unless otherwise specified).

__all__ = ['logger', 'load_json_file', 'dump_json_file']

# Cell

import logging
import time
from dataclasses import dataclass
from typing import *
from pathlib import Path
import json

from .utils import sort_dict, measure_time
from .coco import *


# Cell
logger = logging.getLogger()

# Cell

def load_json_file(annotations_json: Union[str, Path], *, kind: str = "object_detection") -> CocoDataset:
    from_dict_function = get_dataset_class(kind).from_dict

    annotations_json = Path(annotations_json)
    logger.info(f"Loading json_file from: {annotations_json}")
    ext = annotations_json.suffix
    if ext != '.json':
        raise ValueError(f'Expect .json file as input, got: {annotations_json}')

    with measure_time() as timer:

        with measure_time() as timer2:
            D = json.loads(annotations_json.read_text())
        logger.info(f"  json file loaded: elapsed {timer2.elapsed}")

        with measure_time() as timer2:
            coco = from_dict_function(D)
        logger.info(f"  dataset constructed: elapsed {timer2.elapsed}")

    logger.info(f"Loaded json_file: elapsed {timer.elapsed}: {coco.to_full_str()}")

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

    with measure_time() as timer:
        raw = sort_dict(to_dict_function(coco))
    logger.info(f"Sorted keys: elapsed {timer.elapsed}")

    logger.info(f"Writing dataset {coco.to_full_str()} to json-file: {annotations_json}")
    with measure_time() as timer:
        annotations_json.parent.mkdir(parents=True, exist_ok=True)
        annotations_json.write_text(json.dumps(raw, indent=indent, ensure_ascii=False))
    logger.info(f"Dataset written to {annotations_json}: elapsed {timer.elapsed}")