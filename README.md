# Photo Organizer

Offline AI photo organizer powered by **CLIP**: classifies people vs. scenery, groups similar
scenes into clusters, and ships with a local web UI for browsing and fixing results.
**Photos never leave your machine — everything runs locally.**

## Features

- **People / scenery classification** — zero-shot via CLIP text prompts, no training data needed.
- **Scene clustering** — visually similar landscapes are grouped into `cluster_XXX` folders.
- **Auto-labeled clusters** — each cluster folder gets a descriptive scene name (e.g. `cluster_003_snow_mountain`).
- **Local web UI** — thumbnail grid, full-size viewer, drag photos between folders, rename clusters.
- **Phone access** — optional LAN mode lets you review photos from your phone on the same Wi-Fi.
- **Embedding cache** — re-clustering is near-instant; images are only encoded once.
- **RAW-aware** — paired RAW/XMP sidecar files follow the JPEG when you move it.

## Files

| File | Purpose |
|---|---|
| `organize_photos.py` | Main pipeline: scan → CLIP encode → people/scenery split → cluster → write out |
| `label_clusters.py` | Auto-names each `cluster_XXX` folder from its dominant scene |
| `app.py` + `static/` | Local web app: browse thumbnails, move photos, rename folders |
| `start_photo_app.bat` | Launch the web app (localhost only) |
| `start_photo_app_lan.bat` | Launch the web app on your LAN (phone access; allow firewall on first run) |
| `label_clusters.bat` | Run cluster auto-naming |
| `make_test_photos*.py` | Generate synthetic photos for testing the pipeline |

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Requires Python 3.9+. CLIP weights download automatically on first run (GPU optional — CPU works,
just slower).

## Usage

**Organize a folder of photos**

```bash
python organize_photos.py --input "<photo folder>" --output "<output folder>" --region-mode
```

Useful flags:

| Flag | Effect |
|---|---|
| `--min-cluster-size 8` | Coarser clustering (fewer, larger groups) |
| `--move` | Move files instead of copying |
| `--limit 100` | Dry-run on a small subset |

Already-encoded photos are cached in `.embeddings_cache.pkl`, so re-running the clustering step
takes seconds.

**Browse and fix results**

1. Run `start_photo_app.bat` — the browser opens at `http://127.0.0.1:8765`.
2. For phone access, run `start_photo_app_lan.bat` instead and open the printed
   `http://192.168.x.x:8765` on your phone.

In the web UI:

- Pick **region → folder** from the two dropdowns, then browse the thumbnail grid; click any photo
  for the full-size viewer (click left/right edges to page through).
- **Select** → tick photos → **Move to…** to relocate misclassified shots (paired RAW/XMP follow along).
- **Rename folder** to give the current cluster a meaningful name.

## Output structure

```
<output root>/
  <region>/people/              portraits
  <region>/uncertain/           low-confidence, needs a human look
  <region>/scenery/cluster_XXX/ one visually similar group
  <region>/scenery/misc/        leftover scenery
  manifest.json                 per-photo record: original path → destination
```

## License

MIT
