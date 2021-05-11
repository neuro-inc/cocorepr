# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/95_cli.ipynb (unless otherwise specified).

__all__ = ['logger', 'get_parser', 'main']

# Cell

import argparse
import logging
from pathlib import Path
import random

from .coco import merge_datasets, cut_annotations_per_category
from .json_file import *
from .json_tree import *
from .crop_tree import *

# Cell
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger()

# Cell

def get_parser():
    parser = argparse.ArgumentParser(
        description="Tool for converting datasets in COCO format between different representations"
    )
    parser.add_argument("--in_json_file", type=Path, nargs="*", default=[],
                        help=(
                            "Path to one or multiple json files storing COCO dataset "
                            "in `json_file` representation (all json-based datasets will be merged)."
                        ))

    parser.add_argument("--in_json_tree", type=Path, nargs="*", default=[],
                        help=(
                            "Path to one or multiple directories storing COCO dataset "
                            "in `json_tree` representation (all json-based datasets will be merged)."
                        ))

    parser.add_argument("--in_crop_tree", type=Path, nargs="*", default=[],
                        help=(
                            "Path to one or multiple directories storing COCO dataset "
                            "in `crop_tree` representation (all crop-based datasets will be merged and will "
                            "overwrite the json-based datasets)."
                       ))

    parser.add_argument("--out_path", type=Path, required=True,
                        help="Path to the output dataset (file or directory: depends on `--out_format`)")

    parser.add_argument("--out_format", choices=['json_file', 'json_tree', 'crop_tree'], required=True)

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--max_crops_per_class", type=int, default=None,
                        help=(
                            "If set, the tool will randomly select up to this number of "
                            "crops (annotations) per each class (category) and drop the others."),
                       )

    parser.add_argument("--overwrite", action='store_true',
                        help="If set, will delete the output file/directory before dumping the result dataset.")
    parser.add_argument("--indent", default=4,
                        type=lambda x: int(x) if str(x).lower() not in ('none', 'null', '~') else None,
                        help="Indentation in the output json files.")

    parser.add_argument("--debug", action='store_true')

    return parser

# Cell

def main(args=None):
    args = args or get_parser().parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f'Arguments: {args}')

    in_json_tree_list = args.in_json_tree
    in_json_file_list = args.in_json_file
    in_crop_tree_list = args.in_crop_tree

    seed = args.seed
    max_crops_per_class = args.max_crops_per_class

    out_path = args.out_path
    out_format = args.out_format
    overwrite = args.overwrite
    indent = args.indent

    random.seed(args.seed)

    coco = None
    coco_count = 0
    for in_json_tree in in_json_tree_list:
        coco = merge_datasets(coco, load_json_tree(in_json_tree))
        coco_count += 1
    for in_json_file in in_json_file_list:
        coco = merge_datasets(coco, load_json_file(in_json_file))
        coco_count += 1

    if coco is None:
        raise ValueError(f'Not found base dataset, please specify either of: '
                         '--in_json_tree / --in_json_file (multiple arguments allowed)')
    if coco_count > 1:
        logger.info(f'Total loaded json dataset: {coco.to_full_str()}')

    coco_crop = None
    coco_crop_count = 0
    for in_crop_tree in in_crop_tree_list:
        coco_crop = merge_datasets(coco_crop, load_crop_tree(in_crop_tree, coco))
        coco_crop_count += 1
    if coco_crop is not None:
        if coco_crop_count > 1:
            logger.info(f'Total loaded crop-tree dataset: {coco_crop.to_full_str()}')
        logger.info('Using coco_crop dataset.S')
        coco = coco_crop

    if max_crops_per_class:
        logger.info(f'Cutting off crops up to {max_crops_per_class} per class, random seed={seed}')
        coco = cut_annotations_per_category(coco, max_crops_per_class)
        logger.info(f'After cutting off: {coco.to_full_str()}')

    if out_format == 'json_file':
        dump_fun = dump_json_file
    elif out_format == 'json_tree':
        dump_fun = dump_json_tree
    elif out_format == 'crop_tree':
        dump_fun = dump_crop_tree
    else:
        raise ValueError(out_format)
    dump_fun(coco, out_path, skip_nulls=True, overwrite=overwrite, indent=indent)

    details = f': {[p.name for p in out_path.iterdir()]}' if out_path.is_dir() else ''
    logger.info(f'[+] Success: {out_format} dumped to {out_path}' + details)