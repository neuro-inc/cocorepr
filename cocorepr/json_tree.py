# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_json_tree.ipynb (unless otherwise specified).

__all__ = ['logger', 'load_json_tree', 'dump_json_tree']

# Cell

import logging
from typing import *
from pathlib import Path
import json
import shutil

from .utils import sort_dict, measure_time
from .coco import *

# Cell
logger = logging.getLogger()

# Cell

def load_json_tree(tree_dir: Union[str, Path], *, kind: str = "object_detection") -> CocoDataset:
    dataset_class = get_dataset_class(kind)
    from_dict_function = dataset_class.from_dict

    tree_dir = Path(tree_dir)
    logger.info(f"Loading json_tree from dir: {tree_dir}")
    if not tree_dir.is_dir():
        raise ValueError(f"Source json_tree dir not found: {tree_dir}")

    D = {}
    with measure_time() as timer:

        with measure_time() as timer2:

            for el_name in dataset_class.get_collective_elements():
                el_dir = tree_dir / el_name
                if not el_dir.is_dir():
                    logger.debug(f'Chunks dir not found: {el_dir}')
                    el_list = []
                else:
                    el_list = [json.loads(f.read_text()) for f in el_dir.glob('*.json')]
                logger.debug(f'Loaded {len(el_list)} json chunks from {el_dir}')
                D[el_name] = el_list

            for el_name in dataset_class.get_non_collective_elements():
                el_file = tree_dir / f'{el_name}.json'
                if not el_file.is_file():
                    logger.debug(f'Chunks file not found: {el_file}')
                    el = {}
                else:
                    el = json.loads(el_file.read_text())
                logger.debug(f'Loaded single-file json chunk {el_file}')
                D[el_name] = el

        logger.info(f"- json files loaded: elapsed {timer2.elapsed}")

        with measure_time() as timer2:
            coco = dataset_class.from_dict(D)
        logger.info(f"- dataset constructed: elapsed {timer2.elapsed}")

    logger.info(f"Loaded from json_tree: {coco.to_full_str()}")
    return coco

# Cell

def dump_json_tree(
    coco: CocoDataset,
    target_dir: Union[str, Path],
    *,
    kind: str = 'object_detection',
    skip_nulls: bool = True,
    overwrite: bool = False,
    indent: Optional[int] = 4,
) -> None:
    dataset_class = get_dataset_class(kind)
    if skip_nulls:
        to_dict_function = dataset_class.to_dict_skip_nulls
    else:
        to_dict_function = dataset_class.to_dict

    target_dir = Path(target_dir)
    raw = to_dict_function(coco)
    logger.info(f"Dumping json_tree to dir: {target_dir}")

    if overwrite:
        if target_dir.is_dir():
            logger.warning(f'Destination dir exists and will be overwritten: {target_dir}')
    elif target_dir.is_dir():
        raise ValueError(f"Destination json tree dir already exists: {target_dir}")

    if target_dir.is_dir():
        logger.info(f'Deleting old target tree directory {target_dir}')
        shutil.rmtree(str(target_dir))

    target_dir.mkdir(parents=True)

    with measure_time() as timer:
        # TODO: rename cat -> el_kind
        for cat in dataset_class.get_collective_elements():
            el_dir = target_dir / cat
            if not raw.get(cat):
                logger.debug(f'Skipping empty category {el_dir}')
                continue
            el_dir.mkdir()
            for el in raw[cat]:
                el_file = el_dir / f'{el["id"]}.json'
                el = sort_dict(el)
                el_file.write_text(json.dumps(el, indent=indent, ensure_ascii=False))
            logger.debug(f'Written {len(raw[cat])} elements to {el_dir}')

        for cat in dataset_class.get_non_collective_elements():
            el_dir = target_dir / cat
            el_dir.mkdir()
            el_file = target_dir / f'{cat}.json'
            el = raw[cat]
            el = sort_dict(el)
            el_file.write_text(json.dumps(el, indent=indent, ensure_ascii=False))
            logger.debug(f'Written single element to {el_dir}')
    logger.info(f'Dataset written to {target_dir}: elapsed {timer.elapsed}')