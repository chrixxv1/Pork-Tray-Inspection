# AI-use log

| Date | Tool | What it was used for | How I checked it |
|---|---|---|---|
| 2026-10-24 | Claude | Dataset audit (class counts, split sizes, near-duplicate check); generated a reference implementation of the whole project | Rebuilding each part myself and comparing; will re-run audit checks in notebooks/01_explore.ipynb |
| 2026-10-24 | Claude | Step-by-step guidance on setting up Colab, Drive and GitHub | Followed and tested each step |
| 2026-10-24 | Claude | Optimised user-written code to inspect the COCO JSON structure, count images/annotations/classes per split, validate annotations (area, bounds, ID links) and draw boxes on example images | Ran all cells myself; confirmed counts match the Roboflow README; worked through the JSON fields until I could explain each one |
| 2026-10-24 | Claude | Debugged Colab and Git errors (Drive paths, %%writefile filename, push URL and token variable) | Fixed each error myself following the explanation; understood the cause of each |
