"""Shared pytest fixtures."""

import json

import pytest
from PIL import Image


@pytest.fixture
def tiny_coco(tmp_path):
    """Write a 3-image COCO split (one PASS, two FAIL) and return its folder.

    Image 0 has no annotations (PASS), image 1 has one 'unsealed' box and
    image 2 has an 'unsealed' and a 'wrinkle' box (both FAIL).
    """
    images = []
    for i in range(3):
        name = f"frame_{i}.jpg"
        Image.new("RGB", (720, 540), (40, 40, 40)).save(tmp_path / name)
        images.append({"id": i, "file_name": name, "height": 540, "width": 720,
                       "extra": {"name": f"src_{i}.jpg"}})
    coco = {
        "images": images,
        "categories": [{"id": 4, "name": "unsealed"}, {"id": 5, "name": "wrinkle"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "category_id": 4, "bbox": [5, 5, 10, 10]},
            {"id": 3, "image_id": 2, "category_id": 5, "bbox": [50, 50, 10, 10]},
        ],
    }
    (tmp_path / "_annotations.coco.json").write_text(json.dumps(coco))
    return tmp_path
