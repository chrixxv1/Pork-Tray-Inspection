"""Tests of COCO loading, label definition and input validation."""

import json

import pytest

from porkqc.data import load_split, summarise


def test_labels_follow_annotations(tiny_coco):
    records = load_split(tiny_coco)
    assert [r.label for r in records] == [0, 1, 1]
    assert records[0].categories == ()
    assert records[2].categories == ("unsealed", "wrinkle")


def test_source_name_comes_from_extra(tiny_coco):
    records = load_split(tiny_coco)
    assert records[1].source_name == "src_1.jpg"


def test_summary(tiny_coco):
    s = summarise(load_split(tiny_coco))
    assert s["n_images"] == 3
    assert s["fail_fraction"] == pytest.approx(2 / 3)
    assert s["images_per_class"] == {"unsealed": 2, "wrinkle": 1}


def test_missing_annotation_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path)


def test_missing_image_is_reported(tiny_coco):
    (tiny_coco / "frame_0.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="frame_0"):
        load_split(tiny_coco)


def _edit_coco(folder, change):
    """Load the fixture's JSON, apply `change` to it, and save it back."""
    path = folder / "_annotations.coco.json"
    coco = json.loads(path.read_text())
    change(coco)
    path.write_text(json.dumps(coco))


def test_unknown_category_is_reported(tiny_coco):
    _edit_coco(tiny_coco, lambda c: c["annotations"][0].update(category_id=99))
    with pytest.raises(ValueError, match="unknown category"):
        load_split(tiny_coco)


def test_unknown_image_is_reported(tiny_coco):
    _edit_coco(tiny_coco, lambda c: c["annotations"][0].update(image_id=99))
    with pytest.raises(ValueError, match="unknown image"):
        load_split(tiny_coco)


def test_missing_key_is_reported(tiny_coco):
    _edit_coco(tiny_coco, lambda c: c.pop("categories"))
    with pytest.raises(ValueError, match="categories"):
        load_split(tiny_coco)
