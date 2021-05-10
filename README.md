# Title



Cocorepr
========

A tool to convert COCO datasets between representations.
{% include note.html content='for now, only Object Detection COCO is supported' %}
## Installation

```bash
$ pip install -U cocorepr
```

## Basic usage

```
$ cocorepr
usage: cocorepr [-h] [--in_json_tree [IN_JSON_TREE [IN_JSON_TREE ...]]]
                [--in_json_file [IN_JSON_FILE [IN_JSON_FILE ...]]]
                [--in_crop_tree [IN_CROP_TREE [IN_CROP_TREE ...]]] --out_path
                OUT_PATH --out_format {json_file,json_tree,crop_tree}
                [--overwrite] [--indent INDENT] [--debug]
cocorepr: error: the following arguments are required: --out_path, --out_format
```

This tool converts a dataset between three formats:
- json file (a single json file) - common ML format,
- json tree (a set of json chunks) - suitable for Git,
- crop tree (a set of png crops of the object detection annotations) - used for cleaning the object detection dataset.


While json-based formats are self-contained, crop-based format needs at least one json path in order to reconstruct the dataset:
```
$ cocorepr \
    --in_crop_tree /path/to/tree  \
    --out_path /tmp/crop_tree \
    --out_format crop_tree
INFO: Arguments: Namespace(debug=False, in_crop_tree=[PosixPath('/path/to/tree')], in_json_file=[], in_json_tree=[], indent=4, out_format='crop_tree', out_path=PosixPath('/tmp/crop_tree'), overwrite=False)
Traceback (most recent call last):
  File "/home/ay/.pyenv/versions/3.7.6/bin/cocorepr", line 33, in <module>
    sys.exit(load_entry_point('cocorepr', 'console_scripts', 'cocorepr')())
  File "/plain/github/nm/cocorepr/cocorepr/main.py", line 66, in main
    raise ValueError(f'Not found base dataset, please specify either of: '
ValueError: Not found base dataset, please specify either of: --in_json_tree / --in_json_file (multiple arguments allowed)
```


Options `--in_json_tree`, `--in_json_file` and `--in_crop_tree` expect 1 or more path to the specified dataset representation.
If multiple values were passed, the datasets will be merged (enforcing all the elements to have unique `id` fields).
```
$ cocorepr \
    --in_json_file /tmp/json_file/file1.json /tmp/json_file/file2.json \
    --in_json_tree /tmp/json_tree/dir1 /tmp/json_file/dir2 /tmp/json_file/dir3 \
    --in_crop_tree /tmp/crop_tree/dir1 /tmp/crop_tree/dir2 \
    --out_path /tmp/json_tree \
    --out_format json_tree
```

The command above will load `json_file` dataset from `/tmp/json_file/file1.json`, then load `/tmp/json_file/file2.json` and merge it with the first one, then load the `json_tree` from `/tmp/json_tree/dir1` and merge it with the previous result, etc.
Then it'll load the `crop_tree` from `/tmp/crop_tree/dir1` using meta-info from the previously constructed dataset and merge it with `/tmp/crop_tree/dir2`.
The result will be written in form of `json_tree` to `/tmp/json_tree` (if directory exists, the tool will fail unless the `--overwrite` is specified).


## Motivation

This tool was born in [Neu.ro](https://neu.ro) when we worked on an ML project for a client who needed a system that would process photos, detect objects and then classify them by one a large number of classes. The client had large volumes of data, but the data was very noisy.

Roughly, our solution comprised two models:
1. Object Detection (`OD`) model: trained on a dataset and finding generic objects (similar to COCO: bottle, laptop, bus),
2. Object Classification (`CL`) model: fine-tuned on the client's domain (for example: which exactly mark of the bottle, which type of laptop).

While the first model could be generated on a generic dataset, the second problem required large amount of work with the client on cleaning the noisy data and preparing a fine-tuned classification dataset.

For historical reasons, both datasets were collected, cleaned and stored in COCO format. Hopefully, we didn't need to store image blobs -- the client's API enforced their availability and immutability, therefore we could store only image URL and some other metadata (`coco_url` and `id`, other fields are optional):

```json
{
    "id": 49428,  // image ID
    "coco_url": "http://images.cocodataset.org/train2017/000000049428.jpg",  // URL of the immutable image blob
    // "license": 6,
    // "file_name": "000000049428.jpg",
    // "height": 427,
    // "width": 640,
    // "date_captured": "2013-11-15 04:30:29",
    // "flickr_url": "http://farm7.staticflickr.com/6014/5923365195_bee5603371_z.jpg"
},
```

Though COCO format is native fine for OD datasets, it might be bulky for CL datasets, which are concerned on the class of annotations, not images:
```json
{
    "id": 124710,  // annotation ID
    "image_id": 140006,  // image ID in the section "images"
    "category_id": 2,  // class ID in the section "categories"
    "bbox": [496.52, 125.94, 143.48, 113.54],  // crop coordinates in pixels: [x,y,w,h] (from top-left, x=horizontal)
}
```

In order to train a CL model, we want to have a certain number of "clean" crops per each class (by *crop* we call a small picture cropped from given image using coordinates of given annotation). In order to facilitate the manual process of choosing the clean crops, we would like them to be sorted into directories grouping them into classes (categories). After the cleaning, we would like to reconstruct this subset of COCO dataset, register it in Git and then use it to train the model.
Here comes `cocorepr`, which was created to automate these conversions between different representations of a COCO dataset.

Below you can find the detailed discussion of the COCO dataset representations.

---

```python
## Representations of COCO dataset
```

### Json file

This is a regular format for a COCO dataset: all the annotations are stored in a single json file:

```json
$ cat examples/coco_chunk/json_file/instances_train2017_chunk3x2.json
{
    "licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
        ...
    ],
    "info": {
        "description": "COCO 2017 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2017,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    },
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person"
        },
        ...
    ],
    "images": [
        {
            "license": 6,
            "file_name": "000000049428.jpg",
            "coco_url": "http://images.cocodataset.org/train2017/000000049428.jpg",
            "height": 427,
            "width": 640,
            "date_captured": "2013-11-15 04:30:29",
            "flickr_url": "http://farm7.staticflickr.com/6014/5923365195_bee5603371_z.jpg",
            "id": 49428
        },
        ...
    ],
    "annotations": [
        {
            "image_id": 140006,
            "bbox": [
                496.52,
                125.94,
                143.48,
                113.54
            ],
            "category_id": 2,
            "id": 124710
        },
        ...
    ]
}
```

This format is used by many ML frameworks as input format, but usually the json tree file is too big to be stored in a Git repository (over 50M), therefore we either need to store it under Git LFS (which does not show the diff, only the hash), or to use another representation that are better adapted for work with Git.

### Json tree

This format makes the dataset suitable for Git: it stores each element in a separate json chunk, thus enabling Git to do the diff at the level of individual chunks.

```
$ cocorepr \
    --in_json_file examples/coco_chunk/json_file/instances_train2017_chunk3x2.json \
    --out_path $TMP \
    --out_format json_tree  # --overwrite
INFO:root:Arguments: Namespace(in_crop_tree_path=None, in_json_path=PosixPath('examples/coco_chunk/json_file/instances_train2017_chunk3x2.json'), out_format='json_tree', out_path=PosixPath('/tmp/json_tree'), overwrite=False)
INFO:root:Loading json file from file: examples/coco_chunk/json_file/instances_train2017_chunk3x2.json
INFO:root:Loaded: images=6, annotations=6, categories=3
INFO:root:Dumping json tree to dir: /tmp/json_tree
INFO:root:[+] Success: json_tree dumped to /tmp/json_tree: ['info.json', 'info', 'categories', 'annotations', 'licenses', 'images']

$ tree /tmp/json_tree
/tmp/json_tree
├── annotations
│   ├── 124710.json
│   ├── 124713.json
│   ├── 131774.json
│   ├── 131812.json
│   ├── 183020.json
│   └── 183030.json
├── categories
│   ├── 1.json
│   ├── 2.json
│   └── 3.json
├── images
│   ├── 117891.json
│   ├── 140006.json
│   ├── 289949.json
│   ├── 49428.json
│   ├── 537548.json
│   └── 71345.json
├── info
├── info.json
└── licenses
    ├── 1.json
    ├── 2.json
    ├── 3.json
    ├── 4.json
    ├── 5.json
    ├── 6.json
    ├── 7.json
    └── 8.json

5 directories, 24 files
```

### Crop tree

This format is used to facilitate the process of manual cleaning the CL dataset: the directory `crop` contains the list of classes named as `{sanitized-class-name}--{class-id}` so that the classes that have similar name (for example the classes of the cars `Bugatti Veyron EB 16.4` and `Bugatti Veyron 16.4 Grand Sport` will be named as `Bugatti_Veyron_EB_16_4--103209` and `Bugatti_Veyron_16_4_Grand_Sport--376319`, which makes sense since the directories are usually sorted alphabetically). The human then goes through the pictures of crops, deletes the "dirty" ones and makes sure that each class contains enough of "clean" crops. Then, we can reconstruct the dataset in the json tree representation and register it in Git.

```bash
$ cocorepr \
    --in_json_file examples/coco_chunk/json_file/instances_train2017_chunk3x2.json \
    --out_path /tmp/crop_tree \
    --out_format crop_tree
INFO:root:Arguments: Namespace(in_crop_tree_path=None, in_json_path=PosixPath('examples/coco_chunk/json_file/instances_train2017_chunk3x2.json'), indent=4, out_format='crop_tree', out_path=PosixPath('/tmp/crop_tree'), overwrite=False)
INFO:root:Loading json file from file: examples/coco_chunk/json_file/instances_train2017_chunk3x2.json
INFO:root:Loaded: images=6, annotations=6, categories=3
INFO:root:Detected input dataset type: json_file: examples/coco_chunk/json_file/instances_train2017_chunk3x2.json
INFO:root:Dumping crop tree to dir: /tmp/crop_tree
Processing images: 100%|                                           | 6/6 [00:03<00:00,  1.60it/s]
INFO:root:[+] Success: crop_tree dumped to /tmp/crop_tree: ['crops', 'images']

$ tree /tmp/crop_tree
/tmp/crop_tree
├── crops
│   ├── bicycle--2
│   │   ├── 124710.png
│   │   └── 124713.png
│   ├── car--3
│   │   ├── 131774.png
│   │   └── 131812.png
│   └── person--1
│       ├── 183020.png
│       └── 183030.png
└── images
    ├── 000000049428.jpg
    ├── 000000071345.jpg
    ├── 000000117891.jpg
    ├── 000000140006.jpg
    ├── 000000289949.jpg
    └── 000000537548.jpg

5 directories, 12 files
```

Now, this tree can be manually cleaned by a human ("dirty" crops deleted) and we'll be able to re-construct the dataset.

## Showcase: single iteration of the dataset cleaning process

Our setup:
- Our dataset stored in git repository `/project/my-dataset` in the `json_tree` representation. This dataset suffers from incompleteness: some categories lack "clean" annotations.
- The customer has provided us with additional data as two `json_file`s: `/inputs/annotations-new-1.json` and `/inputs/annotations-new-2.json`.
- We would like to merge these two datasets into a `crop_tree` representation, clean it manually, and then re-construct a new dataset and save it in-place in our git repository.

*Step 1*: merge datasets `json_tree` + `json_file`x2 -> `crop_tree`:
```bash
cocorepr \
    --in_json_tree /project/my-dataset \
    --in_json_file /inputs/annotations-new-1.json /inputs/annotations-new-2.json \
    --out_path /temp/my-dataset-crops \
    --out_format crop_tree \
    --overwrite \
    --debug
ls /temp/my-dataset-crops
```

*Step 2*: manually clean the `crop_tree` in `/temp/my-dataset-crops`

*Step 3*: re-construct the cleaned dataset:

```bash
# first, verify that your original dataset has no uncommitted changes (they'll be lost)
cd /project/my-dataset
git diff-index --quiet HEAD

cocorepr \
    --in_crop_tree /temp/my-dataset-crops \
    --in_json_tree /project/my-dataset \
    --out_path /project/my-dataset \
    --out_format json_tree \
    --overwrite \
    --debug
```

Now you can commit the changes of your dataset `/project/my-dataset`.

