# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_crop_tree.ipynb (unless otherwise specified).

__all__ = ['load_crop_tree', 'dump_crop_tree']

# Cell

import json
import shutil
import logging
from collections import defaultdict
from dataclasses import dataclass, Field
from typing import *
from pathlib import Path

from .utils import sort_dict, measure_time, read_image, write_image, cut_bbox
from .coco import *

# Cell

def load_crop_tree(
    source_dir: Union[str, Path],
    base_coco: CocoDataset,
    *,
    kind: str = "object_detection",
) -> CocoDataset:
    """ Load modified set of crops from `{path}/crops` and use it
        to filter out the annotations in `base_coco`.
    """
    dataset_class = get_dataset_class(kind)

    source_dir = Path(source_dir)
    logger.info(f"Loading crop_tree from dir: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source dir not found: {source_dir}")

    crops_dir = source_dir / 'crops'
    if not crops_dir.exists():
        raise ValueError(f'Source crops dir not found: {crops_dir}')

    catid2cat = {cat.id: cat for cat in base_coco.categories}
    imgid2img = {img.id: img for img in base_coco.images}
    annid2ann = {ann.id: ann for ann in base_coco.annotations}
    annid2imgid = {ann.id: ann.image_id for ann in base_coco.annotations}

    res_cats = {}
    res_imgs = {}
    res_anns = {}

    with measure_time() as timer1:
        for count1, ann_dir in enumerate(crops_dir.iterdir(), 1):
            cat_id = int(ann_dir.name.split('--')[-1])
            cat = catid2cat[cat_id]

            with measure_time() as timer2:
                for count2, ann_file in enumerate(ann_dir.glob('*.png'), 1):
                    ann_id = int(ann_file.stem)
                    ann = annid2ann[ann_id]
                    img_id = annid2imgid[ann_id]
                    img = imgid2img[img_id]

                    res_cats[cat.id] = cat
                    res_imgs[img.id] = img
                    res_anns[ann.id] = ann
            logger.info(f'  loaded {count2} crops from {ann_dir}: elapsed {timer2.elapsed}')
        logger.info(f'Loaded from {count1} crop directories: elapsed {timer1.elapsed}')

    with measure_time() as timer:
        D = {
            **base_coco.to_dict(),
            'images': list(res_imgs.values()),
            'annotations': list(res_anns.values()),
            'categories': list(res_cats.values()),
        }
    logger.info(f'Dataset dict constructed: elapsed {timer.elapsed}')

    with measure_time() as timer:
        coco = dataset_class.from_dict(D)
    logger.info(f'Dataset object constructed: elapsed {timer.elapsed}: {coco.to_full_str()}')

    return coco

# Cell

def dump_crop_tree(
    coco: CocoDataset,
    target_dir: Union[str, Path],
    *,
    kind: str = 'object_detection',
    skip_nulls: bool = False,
    overwrite: bool = False,
    indent: Optional[int] = 4,
) -> None:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        logger.warning("Could not import tqdm, please run 'pip install tqdm'")
        def tqdm(it, *args, **kwargs):
            yield from it

    dataset_class = get_dataset_class(kind)
    if skip_nulls:
        to_dict_function = dataset_class.to_dict_skip_nulls
    else:
        to_dict_function = dataset_class.to_dict

    target_dir = Path(target_dir)
    logger.info(f"Dumping crop_tree to dir: {target_dir}")

    if overwrite:
        if target_dir.is_dir():
            logger.warning(f'Destination and will be overwritten: {target_dir}')
    elif target_dir.is_dir():
        raise ValueError(f"Destination json tree dir already exists: {target_dir}")

    if target_dir.is_dir():
        logger.info(f'Deleting old target directory {target_dir}')
        shutil.rmtree(str(target_dir))

    target_dir.mkdir(parents=True)
    catid2cat = {cat.id: cat for cat in coco.categories}

    imgid2img = {img.id: img for img in coco.images}
    imgid2anns = defaultdict(list)
    for ann in coco.annotations:
        imgid2anns[ann.image_id].append(ann)

    images_dir = target_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    crops_dir = target_dir / 'crops'
    crops_dir.mkdir(exist_ok=True)

    anns_failed = []
    anns_failed_file = crops_dir / 'crops_failed.ndjson'

    with measure_time() as timer:
        for imgid, anns in tqdm(imgid2anns.items(), desc='Processing images'):
            img = imgid2img[imgid]
            assert img.file_name, f'Empty file name for img: {img}'
            image_file = images_dir / img.file_name
            image = read_image(image_file, download_url=img.coco_url)

            for ann in anns:
                cat = catid2cat[ann.category_id]
                cat_dir = crops_dir / cat.get_dir_name()
                cat_dir.mkdir(exist_ok=True)

                ann_file = cat_dir / f'{ann.id}.png'
                box = cut_bbox(image, ann.bbox)
                try:
                    write_image(box, ann_file)
                except ValueError as e:
                    logger.error(e)
                    anns_failed.append(ann)
                    with anns_failed_file.open('a') as f:
                        f.write(json.dumps(ann.to_dict(), ensure_ascii=False) + '\n')
    logger.info(f'Crops written to {crops_dir}: elapsed {timer.elapsed}')

    if anns_failed:
        logger.warning(f'Failed to process {len(anns_failed)} crops, see file {anns_failed_file}')