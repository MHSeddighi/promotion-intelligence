# AGENTS.md

## Notebook workflow (user preference)
- Edit Jupyter notebooks in **percent format** (`# %%` cell-marker `.py` files, jupytext style) as the editable source of truth, then convert back to `.ipynb` for execution/outputs. Do not hand-edit `.ipynb` JSON directly.
- Convert with jupytext (installed in `.venv`): `jupytext --to py notebook/xx.ipynb` to create the percent file, `jupytext --to ipynb notebook/xx.py` to convert back, then execute with `jupyter nbconvert --to notebook --execute --inplace` to embed outputs.
