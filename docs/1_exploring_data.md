# Phase 1: Data exploration

Notebook: `notebooks/01_explore.ipynb`

## Steps

1. **Setup.** Mounted Google Drive, moved into the repository folder and unzipped
   the dataset to Colab's local disk (`/content/data`) for faster image reading.
2. **Annotation format.** Inspected the image, annotation and category entries of
   `_annotations.coco.json`. Each `bbox` is `[x, y, width, height]` in pixels,
   measured from the top-left corner of the image with y increasing downwards.
   Annotations link to images through `image_id` and to classes through
   `category_id`. Category 0 is an unused Roboflow placeholder.
3. **Counts.** Counted images, annotations, images without annotations and unique
   source frames per split, and the instances of each defect class.
4. **Annotation checks.** Checked that each box's area equals width × height, that
   every box lies inside the 720 × 540 image, and that every annotation refers to
   an existing image and category.
5. **Visual inspection.** Drew the annotated boxes and class labels on random
   examples of each class and on images without annotations.

## Results

### Split sizes

| Split | Images | Annotations | Images without annotations | Unique source frames |
|---|---|---|---|---|
| train | 3349 | 2923 | 684 | 1085 |
| valid | 120 | 121 | 14 | 120 |
| test | 80 | 79 | 12 | 80 |

### Defect instances per class

| Class | Train | Valid | Test |
|---|---|---|---|
| packaging-error | 1712 | 34 | 23 |
| unsealed | 1055 | 81 | 49 |
| wrinkle | 18 | 0 | 0 |
| loose-meat | 66 | 3 | 3 |
| twisted-meat | 72 | 3 | 4 |

### Annotation checks

All three checks passed for all splits: no area mismatches, no boxes outside the
image and no references to missing images or categories.

## Findings

- **Augmentation is applied to training only.** Train contains about three
  exposure-adjusted copies of each source frame; valid and test contain one
  unmodified image per frame, which is appropriate for evaluation.
- **Fewer source images than stated.** The export contains about 1,285 unique
  source frames across all splits, compared with the approximately 1,955 original
  images described in the project information.
- **Rare classes cannot be evaluated.** Wrinkle does not appear in valid or test,
  and loose meat and twisted meat have only 3–4 instances each, so a single miss
  would change their measured recall by 25–33 %.
- **Most images contain a defect.** 80 % of training images, 88 % of validation
  images and 85 % of test images have at least one annotation. Plain accuracy is
  therefore misleading: labelling every image as defective would score 85 % on test.
- **Small evaluation sets.** Valid (120 images) and test (80 images) are small,
  and test has only 12 images without defects, so metrics will carry wide
  uncertainty.
- **Images are video frames.** File names are numbered frames
  (`frame_with_red_<N>.jpg`), so nearby frames may show the same physical tray in
  different splits.

## Observations from example images for each class

- **Unsealed:** large boxes covering a wide part of the tray, marking where the
  film lid has not sealed. The defect is super subtle.
- **Packaging-error:** small boxes mostly at the tray rim or seal edge, where the meat meets the edge of the tray.
- **Loose meat:** boxes around a single rasher lying out of line with the others,
  often at the end of the tray.
- **Twisted meat:** medium boxes around a rasher that is simply twisted.
- **Wrinkle:** small boxes near the tray edge.
- **Images without annotations:** include trays only partly in view at the frame
  edge, so PASS means "no visible defect in this image" rather than "a complete
  good tray".
- **General:** many trays are cut off at the image edge, trays appear at varying
  angles, and the dark mesh conveyor fills most of each frame. Defects often lie
  close to the image edge.
