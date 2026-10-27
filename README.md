# Pork-Tray-Inspection

MMA3001 Numerical Methods and Machine Learning - individual project, Monash University.

Automated **pass/fail inspection of packaged pork-rasher trays** from overhead RGB images, built from
numerical methods: HOG features, PCA via the singular value decomposition, and L2-regularised logistic
regression solved by Newton's method. Gradient descent, the randomized SVD, pixel features and a
neural network are compared as alternatives.

```
image (720 x 540 RGB) -> HOG (5040) -> PCA via SVD (k = 50) -> logistic regression (Newton) -> P(FAIL) -> threshold
```

## Engineering problem and scope

Sealed trays leave a packing line faster than operators can inspect them consistently. A tray with an
unsealed film or misplaced meat can compromise food safety and shelf life. This project supports
automated inspection by flagging suspect trays for a manual check. It addresses the **tray-level
decision only**: it does not localise defects, identify their type or estimate their severity, because
the rarer defect classes have too few validation and test examples (3-4 each, and none for wrinkle)
to evaluate per-class detection.

## Inputs and outputs

| | Description |
|---|---|
| **Input** | One overhead RGB image of a tray, JPEG, 720 x 540 px, 8-bit (0-255). Other sizes are processed but raise a warning, because they lie outside the validated domain. |
| **Training labels** | COCO bounding boxes; an image is FAIL (1) if it has any annotation, otherwise PASS (0). |
| **Output 1** | P(FAIL), a probability in [0, 1]. |
| **Output 2** | PASS/FAIL from a threshold of 0.736, chosen on the validation split for 90% recall. FAIL trays are diverted for manual inspection. |
| **Invalid input** | Missing files raise `FileNotFoundError`; malformed COCO files, wrong array shapes, values outside [0, 1], NaN/inf and labels other than 0/1 raise `ValueError`. |

Function-level inputs and outputs are documented in the docstrings and the HTML documentation.

## Installation

**Google Colab (as used for this project).** Open `notebooks/0_setup.ipynb` and follow its instructions:
it mounts Google Drive, clones this repository, unzips the dataset and installs the package.

**Local machine (Python 3.10+):**

```bash
git clone https://github.com/chrixxv1/Pork-Tray-Inspection.git
cd Pork-Tray-Inspection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Data

The dataset is **not** included in this repository. Download *Pork Rasher Error (Packaging) v4* in **COCO**
format from Roboflow Universe (licence CC BY 4.0) and unzip it so that `train/`, `valid/` and `test/`
each contain `_annotations.coco.json`. See `data/README.md`.

## Usage

Run the notebooks in order; each one documents its own steps and expected outputs.

| Notebook | Purpose |
|---|---|
| `0_setup.ipynb` | Colab, Google Drive and GitHub set-up |
| `1_exploring_data.ipynb` | Dataset exploration, annotation checks, scope |
| `2_data_module.ipynb` | Data loader and PASS/FAIL label |
| `3_features.ipynb` | Pixel and HOG features |
| `4_pca.ipynb` | PCA via the SVD |
| `5_logistic.ipynb` | Logistic regression: Newton's method and gradient descent |
| `6_experiment.ipynb` | Grouped cross-validation, model selection, final test evaluation |
| `7_optimisation.ipynb` | Profiling, timing, FLOPs, arithmetic intensity, roofline |
| `8_documentation.ipynb` | Documentation, test report and submission files |

Run the tests and rebuild the documentation:

```bash
pytest --html=reports/test_report.html --self-contained-html
pdoc porkqc -o docs/api --docformat google
```

## Results (test split: 80 images, 68 FAIL and 12 PASS, evaluated once)

Selected by grouped 5-fold cross-validation: HOG features, k = 50 components, lambda = 0.0001
(CV AUC 0.722).

| Metric | Model | Always-FAIL baseline |
|---|---|---|
| AUC (95 % bootstrap CI) | 0.833 (0.65-0.95) | 0.500 |
| Recall | 0.735 (0.63-0.84) | 1.000 |
| Specificity | 0.833 | 0.000 |
| Precision | 0.962 | 0.850 |
| Balanced accuracy | 0.784 | 0.500 |

- Recall by defect type: unsealed 0.84 (n = 49), packaging-error 0.62 (n = 16), twisted-meat 0.00 (n = 4), loose-meat 0.50 (n = 2).
- Neural-network (MLP) alternative: test AUC 0.815, 0.771, 0.749 over three random seeds.
- Robust to exposure changes of -30 % to +30 %: test AUC 0.830-0.836.
- The 90 % recall target on validation gave 74% on test: the threshold does not transfer
  reliably from 106 FAIL validation images.

Performance (Colab CPU; see `docs/7_optimisation.md`):

- Inference takes 30.6 ms per image (33 images/s); reduced-resolution
  JPEG decoding cuts this to 26.8 ms (37 images/s) with no loss of accuracy.
- PCA: the randomized SVD (10 power iterations) takes 3.2 s against 82.0 s for the full SVD,
  with the same subspace.
- Training at k = 400: Newton's method needs 8 iterations (1015 ms);
  gradient descent needs 2090 (8229 ms).

## Verification and testing

57 pytest tests (`tests/`, report in `reports/test_report.html`), including gradient and Hessian
checks against central finite differences, agreement with scikit-learn (PCA and logistic regression),
quadratic convergence of Newton's method, the explicit-Euler stability limit of gradient descent,
overflow-safe sigmoid and loss, grouped cross-validation leakage, and input validation.
They use small synthetic data, so they run without the dataset.

## Repository structure

```
src/porkqc/      package source (data, features, pca, logistic, model, evaluate, selection, perf)
tests/           pytest suite
notebooks/       Colab notebooks for each phase
docs/            phase summaries; docs/api/ holds the HTML API documentation
results/         metrics (JSON), profiling output and figures
reports/         pytest HTML test report
data/            dataset location (not committed)
```

## Licence

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, from Roboflow Universe; not redistributed here.

## AI use

AI assistance (Claude) was used throughout this project. Every use is recorded in `AI_USE_LOG.md`,
and the report contains the AI-use reflection.
