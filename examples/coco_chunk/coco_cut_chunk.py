import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_annotations_path", type=Path, required=True)
    parser.add_argument("--out_annotations_path", type=Path, required=True)
    parser.add_argument("--num_categories", type=int, default=10)
    parser.add_argument("--num_annotations_each_category", type=int, default=10)
    return parser


def main(args):
    in_annotations_path = args.in_annotations_path
    out_annotations_path = args.out_annotations_path
    num_categories = args.num_categories
    num_annotations_each_category = args.num_annotations_each_category

    assert in_annotations_path.exists(), in_annotations_path
    assert not out_annotations_path.exists(), out_annotations_path
    assert num_categories > 0, num_categories
    assert num_annotations_each_category > 0, num_annotations_each_category

    logger.info(f"Loading annotations: {in_annotations_path}")
    d = json.loads(in_annotations_path.read_text())
    logger.info(
        f"Loaded: images={len(d['images'])}, "
        f"annotations={len(d['annotations'])}, "
        f"categories={len(d['categories'])}"
    )

    D = {}

    D["licenses"] = d["licenses"]
    D["info"] = d["info"]

    categories = d["categories"][:num_categories]
    catids = {el["id"]: num_annotations_each_category for el in categories}
    annotations = []
    for ann in d["annotations"]:
        catid = ann["category_id"]
        if catid in catids:
            if catids[catid] > 0:
                annotations.append(ann)
                catids[catid] -= 1

    imgids = {el["image_id"] for el in annotations}
    images = [el for el in d["images"] if el["id"] in imgids]

    D["categories"] = sorted(categories, key=lambda el: el["id"])
    D["images"] = sorted(images, key=lambda el: el["id"])
    D["annotations"] = sorted(annotations, key=lambda el: el["id"])

    logger.info(
        f"New dataset: images={len(D['images'])}, "
        f"annotations={len(D['annotations'])} (unique: {len({el['id'] for el in D['annotations']})}, "
        f"categories={len(D['categories'])}"
    )
    logger.info(f"Writing results to: {out_annotations_path}")
    out_annotations_path.write_text(json.dumps(D, indent=4))

    logger.info("[+] Success")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s - %(levelname)s:  %(message)s"
    )
    logger = logging.getLogger()
    args = get_parser().parse_args()
    logger.info(f"Arguments: {args}")
    main(args)
