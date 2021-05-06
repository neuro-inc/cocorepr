# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_coco.ipynb (unless otherwise specified).

__all__ = ['logger', 'CocoElement', 'CocoInfo', 'CocoLicense', 'CocoImage', 'CocoAnnotation',
           'CocoObjectDetectionAnnotation', 'CocoCategory', 'CocoObjectDetectionCategory', 'CocoDataset',
           'CocoObjectDetectionDataset', 'get_dataset_class', 'MAP_COCO_TYPE_TO_DATASET_CLASS']

# Cell

import logging
from abc import abstractmethod
from datetime import datetime
from dataclasses_json import dataclass_json
from dataclasses import dataclass, fields, asdict, field
from dataclasses_serialization.json import JSONSerializer
from typing import *

from .utils import sanitize_filename

logger = logging.getLogger()

# Type helpers:
X, Y, W, H = Type[int], int, int, int

# Cell

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

# Cell

@dataclass
class CocoLicense(CocoElement):
    id: int
    name: str
    url: Optional[str] = None

    @property
    def collection_name(self):
        return "licenses"

# Cell

@dataclass
class CocoImage(CocoElement):
    id: int
    coco_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    license: Optional[int] = None
    file_name: Optional[str] = None
    flickr_url: Optional[str] = None
    date_captured: Optional[str] = None

    @property
    def collection_name(self):
        return "images"

# Cell

@dataclass
class CocoAnnotation(CocoElement):
    id: int
    image_id: int

@dataclass
class CocoObjectDetectionAnnotation(CocoAnnotation):
    category_id: int
    bbox: Tuple[X, Y, W, H]
    supercategory: Optional[str] = None
    area: Optional[int] = None
    iscrowd: Optional[int] = None

# Cell

@dataclass
class CocoCategory(CocoElement):
    id: int

    @abstractmethod
    def get_alias(self):
        raise NotImplementedError


@dataclass
class CocoObjectDetectionCategory(CocoCategory):
    name: str
    supercategory: Optional[str] = None

    def get_dir_name(self):
        name = sanitize_filename(self.name)
        return f'{name}--{self.id}'

# Cell

@dataclass
class CocoDataset(CocoElement):
    images: List[CocoImage] = field(default_factory=list)
    info: Optional[CocoInfo] = None
    licenses: Optional[List[CocoLicense]] = None

    @classmethod
    def get_non_collective_elements(cls):
        # TODO: rename to get_individual_elements()
        return ['info']

    @classmethod
    def get_collective_elements(cls):
        default_self = cls()
        non_collective = set(cls.get_non_collective_elements())
        return [f.name for f in fields(default_self) if f.name not in non_collective]

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
