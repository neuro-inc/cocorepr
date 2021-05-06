# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_utils.ipynb (unless otherwise specified).

__all__ = ['logger', 'sort_dict', 'sanitize_filename', 'read_image', 'download_image', 'draw_image', 'cut_bbox',
           'write_image']

# Cell

import re
import cv2
import logging
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
from pathlib import Path
from typing import *

logger = logging.getLogger()

# Cell

def sort_dict(D: Dict, sort_key='id') -> Dict:
    assert isinstance(D, dict), (type(D), D)
    return OrderedDict({
        k: (
            sorted(D[k], key=lambda x: x[sort_key])
            if isinstance(D[k], list) \
                and D[k] \
                and isinstance(D[k][0], dict) \
                and (sort_key in D[k][0])
            else D[k]
        )
        for k in sorted(D.keys())
    })


# Cell

def sanitize_filename(s: str, max_len=256) -> str:
    s = re.sub(r'[-\s]+', '_', s)
    s = re.sub(r'[^\w ]', '', s)
    s = s[:max_len] if len(s) > max_len else s
    s = s.strip('_')
    return s

# Cell

def read_image(
    image_path: Union[str, Path],
    download_url: Optional[str] = None,
) -> np.ndarray:
    """ Reads image :image_path: in RGB mode. If image file does not
        exist and :download_url: was specified, downloads the image first.
    """
    if download_url:
        download_image(image_path, download_url)

    assert Path(image_path).is_file(), f'Image not exists: {image_path}'

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Cell

def download_image(image_path: Union[str, Path], download_url: str):
    image_path = Path(image_path)
    if image_path.exists():
        return
    image_path.parent.mkdir(exist_ok=True, parents=True)
    urllib.request.urlretrieve(download_url, str(image_path))

# Cell

def draw_image(image, figsize=(24, 24)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.show()

# Cell

def cut_bbox(image, bbox):
    x, y, w, h = map(int, bbox)
    crop = image[y:(y+h), x:(x+w)]
    return crop

# Cell

def write_image(image, image_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    Path(image_path).parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(image_path), image)