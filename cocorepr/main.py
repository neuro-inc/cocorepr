# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/90_cli.ipynb (unless otherwise specified).

__all__ = ['logger', 'get_parser', 'main']

# Cell

import argparse
import logging
from pathlib import Path

from .coco import merge_datasets
from .json_file import *
from .json_tree import *
from .crop_tree import *

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger()

# Cell

def get_parser():
    parser = argparse.ArgumentParser(
        description="Tool for converting datasets in COCO format between different formats"
    )
    parser.add_argument("--in_json_tree", type=Path, nargs="*", default=[])
    parser.add_argument("--in_json_file", type=Path, nargs="*", default=[])
    parser.add_argument("--in_crop_tree", type=Path, nargs="*", default=[])

    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--out_format", choices=['json_file', 'json_tree', 'crop_tree'], required=True)

    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--indent", default=4,
                        type=lambda x: int(x) if str(x).lower() not in ('none', 'null', '~') else None)
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

    out_path = args.out_path
    out_format = args.out_format
    overwrite = args.overwrite
    indent = args.indent

    coco = None
    for in_json_tree in in_json_tree_list:
        coco = merge_datasets(coco, load_json_tree(in_json_tree))
    for in_json_file in in_json_file_list:
        coco = merge_datasets(coco, load_json_file(in_json_file))

    if coco is None:
        raise ValueError(f'Not found base dataset, please specify either of: '
                         '--in_json_tree / --in_json_file (multiple arguments allowed)')
    logger.info(f'Loaded total json dataset: '
                f'len(annotations)={len(coco.annotations)} '
                f'len(images)={len(coco.images)} '
                f'len(categories)={len(coco.categories)}')

    coco_crop = None
    for in_crop_tree in in_crop_tree_list:
        coco_crop = merge_datasets(coco_crop, load_crop_tree(in_crop_tree, coco))
    if coco_crop is not None:
        logger.info(f'Loaded total coco_crop dataset: '
                    f'len(annotations)={len(coco_crop.annotations)} '
                    f'len(images)={len(coco_crop.images)} '
                    f'len(categories)={len(coco_crop.categories)}')
        logger.info('Using coco_crop dataset as primary')
        coco = coco_crop

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