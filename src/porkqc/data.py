"""Loading of the Roboflow COCO export and definition of the tray-level label.

The dataset provides bounding boxes for five defect classes. This project
reduces them to a single image-level (tray-level) label:

* ``1`` (FAIL) - the image contains at least one defect annotation;
* ``0`` (PASS) - the image contains no defect annotation.

This reduction is a deliberate scoping decision: the validation and test splits
contain only 3-4 instances of the rarer classes, which is too few to evaluate
per-class detection, but the binary label has 200 evaluation images.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: Splits shipped in the Roboflow export.
SPLITS = ("train", "valid", "test")

#: Expected image size (width, height) after Roboflow pre-processing.
EXPECTED_SIZE = (720, 540)


@dataclass(frozen=True)
class ImageRecord:
    """One image and its tray-level label.

    Attributes:
        path: Absolute path to the JPEG file.
        source_name: Name of the original (pre-augmentation) frame. Augmented
            copies of the same frame share this name.
        label: 1 if the image contains any defect annotation, else 0.
        categories: Sorted names of the defect classes annotated in the image.
    """

    path: Path
    source_name: str
    label: int
    categories: tuple[str, ...]


def load_split(split_dir: str | Path) -> list[ImageRecord]:
    """Load one split of the COCO export.

    Args:
        split_dir: Directory containing ``_annotations.coco.json`` and the
            images it references.

    Returns:
        One :class:`ImageRecord` per image, in the order listed in the JSON.

    Raises:
        FileNotFoundError: If the annotation file or a referenced image is
            missing.
        ValueError: If the annotation file is not valid COCO (missing keys,
            annotations referring to unknown images or categories).
    """
    split_dir = Path(split_dir)
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"No COCO annotation file at {ann_path}")

    with ann_path.open() as f:
        coco = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"{ann_path} is missing the '{key}' key")

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    image_ids = {im["id"] for im in coco["images"]}

    cats_per_image: dict[int, set[str]] = {i: set() for i in image_ids}
    for ann in coco["annotations"]:
        if ann["image_id"] not in image_ids:
            raise ValueError(f"Annotation {ann['id']} refers to unknown image {ann['image_id']}")
        if ann["category_id"] not in cat_names:
            raise ValueError(f"Annotation {ann['id']} has unknown category {ann['category_id']}")
        cats_per_image[ann["image_id"]].add(cat_names[ann["category_id"]])

    records = []
    for im in coco["images"]:
        path = (split_dir / im["file_name"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image listed in {ann_path} not found: {path}")
        cats = tuple(sorted(cats_per_image[im["id"]]))
        # Roboflow stores the pre-augmentation file name in extra.name; fall
        # back to file_name for exports that do not include it.
        source = im.get("extra", {}).get("name", im["file_name"])
        records.append(ImageRecord(path, source, int(bool(cats)), cats))
    return records


def load_dataset(root: str | Path) -> dict[str, list[ImageRecord]]:
    """Load the train, valid and test splits from the export root directory.

    Args:
        root: Directory containing ``train/``, ``valid/`` and ``test/``.

    Returns:
        Mapping from split name to its list of records.
    """
    root = Path(root)
    return {s: load_split(root / s) for s in SPLITS}


def summarise(records: list[ImageRecord]) -> dict[str, object]:
    """Summarise a split for reporting and sanity checks.

    Args:
        records: Records of one split.

    Returns:
        Dictionary with the number of images, number of unique source frames,
        fraction labelled FAIL and the number of images containing each class.
    """
    n = len(records)
    class_counts = Counter(c for r in records for c in r.categories)
    return {
        "n_images": n,
        "n_source_frames": len({r.source_name for r in records}),
        "fail_fraction": sum(r.label for r in records) / n if n else float("nan"),
        "images_per_class": dict(sorted(class_counts.items())),
    }
