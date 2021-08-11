# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_coco.ipynb (unless otherwise specified).

__all__ = ['logger', 'CocoElement', 'CocoInfo', 'CocoLicense', 'CocoImage', 'CocoAnnotation',
           'CocoObjectDetectionAnnotation', 'CocoCategory', 'CocoObjectDetectionCategory', 'CocoDataset',
           'CocoObjectDetectionDataset', 'get_dataset_class', 'MAP_COCO_TYPE_TO_DATASET_CLASS', 'merge_datasets',
           'shuffle', 'cut_annotations_per_category', 'remove_invalid_elements']

# Cell

import logging
import random
from abc import abstractmethod
from datetime import datetime
from dataclasses_json import dataclass_json
from dataclasses import dataclass, fields, asdict, field, replace
from typing import *
from pathlib import Path
from pydantic import validate_arguments
from .utils import sanitize_filename

# Type helpers:
X, Y, W, H = Type[int], int, int, int

# Cell
logger = logging.getLogger()

# Cell
@validate_arguments
@dataclass_json
@dataclass
class CocoElement:
    def to_dict_skip_nulls(self):
        """ Same as `self.to_dict()` but does not add those fields
            whose values are `None`.
            WARNING! If you explicitly set a `None` value to a field
                     that has a non-`None` default value, it still
                     won't be dumped and will be deserialized wrongly.
        """
        return asdict(
            self,
            dict_factory=(
                lambda kv: {
                    k: v
                    for k, v in kv
                    if v is not None
                }
            )
        )

    @property
    def collection_name(self) -> str:
        raise NotImplementedError

    def is_valid(self) -> bool:
        raise NotImplementedError

# Cell

@dataclass
class CocoInfo(CocoElement):
    year: Optional[int] = None
    version: Optional[str] = None
    description: Optional[str] = None
    contributor: Optional[str] = None
    url: Optional[str] = None
    date_created: Optional[str] = None

    @property
    def collection_name(self):
        return "info"

    def is_valid(self) -> bool:
        return True  # no restrictions on the format

# Cell

@dataclass
class CocoLicense(CocoElement):
    id: str
    name: str
    url: Optional[str] = None

    @property
    def collection_name(self):
        return "licenses"

    @property
    def id(self):
        return str(self.id)

    def is_valid(self) -> bool:
        return True  # no restrictions on the format

# Cell

@dataclass
class CocoImage(CocoElement):
    id: str
    coco_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    license: Optional[int] = None
    file_name: Optional[str] = None
    flickr_url: Optional[str] = None
    date_captured: Optional[str] = None

    def is_valid(self) -> bool:
        try:
            assert self.id
            assert self.coco_url
            return True
        except:
            return False

    @property
    def collection_name(self):
        return "images"

    @property
    def id(self):
        return str(self.id)

    def get_file_name(self) -> str:
        return self.file_name or Path(self.coco_url).name

# Cell

@dataclass
class CocoAnnotation(CocoElement):
    id: str
    image_id: str

    def is_valid(self) -> bool:
        try:
            assert self.id
            assert self.image_id
            return True
        except:
            return False

    @property
    def id(self):
        return str(self.id)

    @property
    def image_id(self):
        return str(self.image_id)

    def get_file_name(self) -> str:
        return f'{self.id}.png'

@dataclass
class CocoObjectDetectionAnnotation(CocoAnnotation):
    category_id: str
    bbox: Tuple[X, Y, W, H]
    supercategory: Optional[str] = None
    area: Optional[int] = None
    iscrowd: Optional[int] = None

    def is_valid(self) -> bool:
        if not super().is_valid():
            return False
        try:
            assert self.category_id
            x, y, w, h = map(int, self.bbox)
            assert x >= 0, x
            assert y >= 0, y
            assert w >= 0, w
            assert h >= 0, h
            return True
        except:
            return False

    @property
    def category_id(self):
        return str(self.category_id)

# Cell

@dataclass
class CocoCategory(CocoElement):
    id: str

    @abstractmethod
    def get_alias(self):
        raise NotImplementedError

    def is_valid(self) -> bool:
        try:
            assert self.id
            return True
        except:
            return False

    @property
    def id(self):
        return str(self.id)

@dataclass
class CocoObjectDetectionCategory(CocoCategory):
    name: str
    supercategory: Optional[str] = None

    def get_dir_name(self):
        name = sanitize_filename(self.name)
        return f'{name}--{self.id}'

    def is_valid(self) -> bool:
        if not super().is_valid():
            return False
        if not self.name:
            return False
        return True

# Cell

@dataclass
class CocoDataset(CocoElement):
    annotations: List[CocoAnnotation] = field(default_factory=list)
    images: List[CocoImage] = field(default_factory=list)
    info: CocoInfo = CocoInfo()
    licenses: List[CocoLicense] = field(default_factory=list)

    @classmethod
    def get_non_collective_elements(cls):
        # TODO: rename to get_individual_elements()
        return ['info']

    @classmethod
    def get_collective_elements(cls):
        default_self = cls()
        non_collective = set(cls.get_non_collective_elements())
        return sorted([f.name for f in fields(default_self) if f.name not in non_collective])

    def to_full_str(self):
        return (
            f'{self.__class__.__name__}(' + \
            ', '.join(f'{k}={len(getattr(self, k))}' for k in self.get_collective_elements()) + \
            ')'
        )

    def is_valid(self) -> bool:
        return all(
            el.is_valid()
            for el in self.annotations + self.images + [self.info] + self.licenses
        )

# Cell

@dataclass
class CocoObjectDetectionDataset(CocoDataset):
    annotations: List[CocoObjectDetectionAnnotation] = field(default_factory=list)
    categories: List[CocoObjectDetectionCategory] = field(default_factory=list)

# Cell

MAP_COCO_TYPE_TO_DATASET_CLASS = {
    "object_detection": CocoObjectDetectionDataset,
}

def get_dataset_class(coco_kind: str):
    try:
        return MAP_COCO_TYPE_TO_DATASET_CLASS[coco_kind]
    except KeyError:
        raise ValueError(f"Not supported dataset kind: {kind}")

# Cell

def merge_datasets(d1: CocoDataset, d2: CocoDataset) -> CocoDataset:
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    assert isinstance(d1, CocoDataset), (type(d1), d1)
    assert isinstance(d2, CocoDataset), (type(d1), d1)
    t1 = type(d1)
    t2 = type(d2)
    assert t1 == t2, f'Cannot merge datasets: {t1} != {t2}'

    D1 = d1.to_dict()
    D2 = d2.to_dict()
    K1 = set(D1.keys())
    K2 = set(D2.keys())
    assert K1 == K2, f'Cannot merge datasets: {K1} != {K2}'

    res = {}
    for k in K1:
        if isinstance(D1[k], list):
            v1 = {x['id']: x for x in (D1[k] or [])}
            v2 = {x['id']: x for x in (D2[k] or [])}
            v_res = {}
            for i in v1:
                if i in v2 and v1[i] != v2[i]:
                    raise ValueError(f'Invalid "{k}" of id={i}: {v1[i]} != {v2[i]}')
                v_res[i] = v1[i]
            for i in v2:
                if i in v1 and v2[i] != v1[i]:
                    raise ValueError(f'Invalid "{k}" of id={i}: {v2[i]} != {v1[i]}')
                v_res[i] = v2[i]
            res[k] = sorted(v_res.values(), key=lambda x: str(x['id']))
            # we are converting ID to str since sometimes its integer
        else:
            v1 = D1[k] or {}
            v2 = D2[k] or {}
            if not v1:
                res[k] = v2
            elif not v2:
                res[k] = v1
            else:
                assert v1 == v2, f'key={k}: unexpectedly: {v1} != {v2}'
                res[k] = v1

    return t1.from_dict(res)

# Cell
def shuffle(arr):
    return random.sample(arr, k=len(arr))

# Cell
def cut_annotations_per_category(coco: CocoDataset, max_annotations_per_category: int) -> CocoDataset:
    """ Returns a copy of the input dataset where each class (category)
        contains up to `max_crops_per_class` crops (annotations)
    """
    imgid2img = {img.id: img for img in coco.images}
    catid2anns = {cat.id: [] for cat in coco.categories}
    for ann in coco.annotations:
        catid2anns[ann.category_id].append(ann)

    images = {}
    annotations = {}
    for _, anns in catid2anns.items():
        if len(anns) > max_annotations_per_category:
            anns = shuffle(anns)[:max_annotations_per_category]
        for ann in anns:
            annotations[ann.id] = ann
            images[ann.image_id] = imgid2img[ann.image_id]
    coco = replace(coco, annotations=sorted(annotations.values(), key=lambda x: x.id))
    coco = replace(coco, images=sorted(images.values(), key=lambda x: x.id))

    return coco

# Cell
from collections import defaultdict

def remove_invalid_elements(coco: CocoDataset) -> CocoDataset:
    annid2ann = {ann.id: ann for ann in coco.annotations if ann.is_valid()}
    imgid2img = {img.id: img for img in coco.images if img.is_valid()}
    catid2cat = {cat.id: cat for cat in coco.categories if cat.is_valid()}

    imgid2img_used = {}
    catid2cat_used = {}
    annid2ann_used = {}
    for ann in annid2ann.values():
        img = imgid2img.get(ann.image_id)
        cat = catid2cat.get(ann.category_id)
        if img is not None and cat is not None:
            annid2ann_used[ann.id] = ann
            imgid2img_used[img.id] = img
            catid2cat_used[cat.id] = cat

    coco = replace(
        coco,
        annotations=sorted(annid2ann_used.values(), key=lambda x: str(x.id)),
        images=sorted(imgid2img_used.values(), key=lambda x: str(x.id)),
        categories=sorted(catid2cat_used.values(), key=lambda x: str(x.id)),
    )

    # TODO: filter also licenses
    # TODO: filter also get_non_collective_elements (info)
    return coco