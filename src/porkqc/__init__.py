"""porkqc - tray-level pass/fail inspection of packaged pork rashers.

Given an overhead RGB image of a sealed tray, the package estimates the
probability that the tray has a packaging or product-placement defect and
makes a PASS/FAIL decision:

    image (720 x 540 RGB) -> HOG features (5040) -> PCA via the SVD (k = 50)
    -> logistic regression solved by Newton's method -> P(FAIL) -> threshold

Modules:

* `porkqc.data` - COCO loading, the PASS/FAIL label and input validation
* `porkqc.features` - pixel and HOG feature extraction
* `porkqc.pca` - PCA via the thin SVD and the randomized SVD
* `porkqc.logistic` - logistic regression: Newton's method and gradient descent
* `porkqc.model` - the combined PCA + logistic-regression pipeline
* `porkqc.evaluate` - metrics, threshold selection and bootstrap intervals
* `porkqc.selection` - grouped cross-validation for model selection
* `porkqc.perf` - timing, FLOP and arithmetic-intensity estimates, fast decoding
"""

__version__ = "1.0.0"
