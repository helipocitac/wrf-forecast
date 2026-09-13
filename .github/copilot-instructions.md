# WRFWEB project instructions

## Project shape

- This is a static WRF-ARW forecast viewer plus a Python map/image generator.
- `build_run.py` is the generator entry point. It imports `wrf-python`, NumPy, Matplotlib, Cartopy, SciPy, and related scientific Python packages.
- `index.html` is the browser entry point and reads generated metadata and images from `web_data/`.
- `colortables/` contains the `.ct` color tables used by the generator.
- `wrfvystupy/` contains local WRF NetCDF input files. NetCDF files are ignored by Git.
- `web_data/` contains generated WebP frames and `runs.json`; treat it as generated output.

## Runtime conventions

- The generator currently uses the absolute base path `M:\\wrfouts\\WRFWEB` and a hard-coded input file/run ID near the configuration section of `build_run.py`.
- Run the generator from Windows with the repository's scientific Python environment active. It opens the configured WRF NetCDF file, renders all forecast steps in parallel, and updates `web_data/runs.json`.
- Serve the repository root over HTTP when testing the viewer, because `index.html` fetches `web_data/runs.json` and browser file URLs may block those requests.
- Do not regenerate or modify `web_data/` casually: rendered assets are large and may already contain user changes.

## Change and validation guidance

- Preserve existing uncommitted generated-asset changes. Avoid broad formatting or regeneration when changing source code.
- For Python changes, first run a syntax check such as `python -m py_compile build_run.py`; use a full generator run only when the relevant WRF input and dependencies are available.
- For viewer changes, validate with a local static HTTP server and inspect the browser console/network requests. There is no package manifest or automated test suite in the repository.
- Keep user-facing text and existing Czech UI conventions intact unless the task explicitly requests a language change.
