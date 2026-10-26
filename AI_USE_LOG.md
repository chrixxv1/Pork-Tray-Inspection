# AI-use log

| Date | Tool | What it was used for | How I checked it |
|---|---|---|---|
| 2026-10-24 | Claude | Dataset audit (class counts, split sizes, near-duplicate check); generated a reference implementation of the whole project | Rebuilding each part myself and comparing; will re-run audit checks in notebooks/01_explore.ipynb |
| 2026-10-24 | Claude | Step-by-step guidance on setting up Colab, Drive and GitHub | Followed and tested each step |
| 2026-10-24 | Claude | Optimised user-written code to inspect the COCO JSON structure, count images/annotations/classes per split, validate annotations (area, bounds, ID links) and draw boxes on example images | Ran all cells myself; confirmed counts match the Roboflow README; worked through the JSON fields until I could explain each one |
| 2026-10-24 | Claude | Debugged Colab and Git errors (Drive paths, %%writefile filename, push URL and token variable) | Fixed each error myself following the explanation; understood the cause of each |
| 2026-10-25 | Claude | Explained how to write the code for the package configuration, src/porkqc/data.py (COCO loader, PASS/FAIL labels, input validation) and the 8 pytest tests | Ran the loader on the real dataset and checked its counts against my Phase 1 tables; ran the tests (8 passed); asked Claude to explain parts I did not understand |
| 2026-10-25 | Claude | Wrote the code: src/porkqc/features.py (pixel and HOG features, exposure control, caching, input validation), the feature visualisation and the 10 feature tests | Ran the visualisation and timing on the real data; ran all tests (18 passed); asked Claude to explain parts I did not understand |
| 2026-10-26 | Claude | Wrote the PCA Module code: src/porkqc/pca.py (PCA via thin SVD and randomized SVD), feature caching, explained-variance and reconstruction plots, and the 7 PCA tests | Ran the plots on the real data and checked the variance values; ran all tests (25 passed), including agreement with scikit-learn; asked Claude to explain parts I did not understand |
