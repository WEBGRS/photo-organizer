"""
Separate a photo library into people vs scenery, then cluster similar scenery.
Also carries RAW/XMP sidecar files alongside their matching JPEG, and can treat
each immediate subfolder of an input root as its own region for separate output.

Usage:
    python organize_photos.py --input D:\\Photos --output D:\\Organized
    python organize_photos.py --input "D:\\Trip" --output D:\\Organized --region-mode
    python organize_photos.py --input D:\\Photos --output D:\\Organized --min-cluster-size 8 --limit 200
"""
import argparse
import hashlib
import json
import pickle
import shutil
import traceback
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff"}
SIDECAR_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".xmp"}

PERSON_PROMPTS = [
    "a photo of a person",
    "a portrait of a person",
    "a photo of a group of people",
    "a selfie",
    "a close-up photo of a human face",
    "tourists posing for a photo",
]
SCENERY_PROMPTS = [
    "a photo of a landscape",
    "a photo of natural scenery",
    "a photo of mountains or snow peaks",
    "a photo of the sky or clouds",
    "a photo of the sea, a lake or a beach",
    "a photo of a city street, skyline or building",
    "a photo of a forest, rainforest or park",
    "a photo of an old town, temple or historic architecture",
    "a photo of an interior room with no people",
    "a close-up photo of an object or food",
]


def scan_primaries(roots):
    files = []
    for root in roots:
        root = Path(root)
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                files.append(p)
    return sorted(set(files))


def find_sidecars_and_orphans(roots, primaries):
    primary_by_key = {}
    for p in primaries:
        primary_by_key.setdefault((p.parent, p.stem.lower()), p)

    sidecars = {p: [] for p in primaries}
    orphans = []
    for root in roots:
        root = Path(root)
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS or ext not in SIDECAR_EXTS:
                continue
            primary = primary_by_key.get((f.parent, f.stem.lower()))
            if primary is not None:
                sidecars[primary].append(f)
            else:
                orphans.append(f)
    return sidecars, orphans


def region_of(path: Path, roots) -> str:
    for root in roots:
        try:
            rel = path.resolve().relative_to(Path(root).resolve())
        except ValueError:
            continue
        return rel.parts[0] if len(rel.parts) > 1 else "_root"
    return "_root"


def cache_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{st.st_mtime}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache_path: Path, cache: dict):
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def encode_images(paths, model, preprocess, device, cache, cache_path, batch_size=64):
    embeddings = {}
    keys = {p: cache_key(p) for p in paths}
    to_encode = [p for p in paths if keys[p] not in cache]
    for p in paths:
        if keys[p] in cache:
            embeddings[p] = cache[keys[p]]

    for i in tqdm(range(0, len(to_encode), batch_size), desc="Encoding new images"):
        batch_paths = to_encode[i : i + batch_size]
        imgs, valid_paths = [], []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"[unreadable] {p}: {e}")
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu().numpy()
        for p, f in zip(valid_paths, feats):
            embeddings[p] = f
            cache[keys[p]] = f
        save_cache(cache_path, cache)
    return embeddings


def classify_people_vs_scenery(embeddings, model, tokenizer, device):
    with torch.no_grad():
        person_feats = model.encode_text(tokenizer(PERSON_PROMPTS).to(device))
        person_feats = person_feats / person_feats.norm(dim=-1, keepdim=True)
        scenery_feats = model.encode_text(tokenizer(SCENERY_PROMPTS).to(device))
        scenery_feats = scenery_feats / scenery_feats.norm(dim=-1, keepdim=True)
    person_feats = person_feats.cpu().numpy()
    scenery_feats = scenery_feats.cpu().numpy()

    labels = {}
    for p, emb in embeddings.items():
        person_score = float(np.max(emb @ person_feats.T))
        scenery_score = float(np.max(emb @ scenery_feats.T))
        if person_score > scenery_score:
            labels[p] = ("people", person_score - scenery_score)
        else:
            labels[p] = ("scenery", scenery_score - person_score)
    return labels


def cluster_scenery(scenery_paths, embeddings, min_cluster_size):
    if len(scenery_paths) < min_cluster_size:
        return {p: -1 for p in scenery_paths}
    X = np.stack([embeddings[p] for p in scenery_paths])
    n_components = min(50, X.shape[0] - 1, X.shape[1])
    if n_components >= 2:
        X = PCA(n_components=n_components, random_state=0).fit_transform(X)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(X)
    return dict(zip(scenery_paths, labels))


def unique_dest(dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def transfer_with_sidecars(primary, category_dir, sidecars, transfer, manifest, category, errors):
    category_dir.mkdir(parents=True, exist_ok=True)
    try:
        dest = unique_dest(category_dir, primary.name)
        transfer(str(primary), str(dest))
        manifest.append({"src": str(primary), "dest": str(dest), "category": category})
        for sc in sidecars.get(primary, []):
            sc_dest = unique_dest(category_dir, sc.name)
            transfer(str(sc), str(sc_dest))
            manifest.append({"src": str(sc), "dest": str(sc_dest), "category": category})
    except Exception as e:
        errors.append(f"{primary}: {e}\n{traceback.format_exc()}")


def main():
    parser = argparse.ArgumentParser(description="Separate photos into people/scenery and cluster similar scenery.")
    parser.add_argument("--input", action="append", required=True, help="Input folder (repeatable).")
    parser.add_argument("--output", required=True, help="Output root folder.")
    parser.add_argument("--region-mode", action="store_true", help="Treat each immediate subfolder of an input root as its own region; cluster and output separately per region.")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum photos per scenery cluster.")
    parser.add_argument("--uncertain-margin", type=float, default=0.02, help="Below this score margin, goes to /uncertain.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying.")
    parser.add_argument("--model", default="ViT-B-32", help="open_clip model name.")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", help="open_clip pretrained weights tag.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images (for a quick test run).")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    print("Scanning input folders...")
    primaries = scan_primaries(args.input)
    if args.limit:
        primaries = primaries[: args.limit]
    print(f"Found {len(primaries)} primary images.")
    if not primaries:
        print("No images found, exiting.")
        return

    sidecars, orphans = find_sidecars_and_orphans(args.input, primaries)
    n_sidecars = sum(len(v) for v in sidecars.values())
    print(f"Found {n_sidecars} sidecar files (RAW/XMP) and {len(orphans)} orphan RAW/XMP files with no matching image.")

    cache_path = output_root / ".embeddings_cache.pkl"
    cache = load_cache(cache_path)
    embeddings = encode_images(primaries, model, preprocess, device, cache, cache_path)

    unreadable = [p for p in primaries if p not in embeddings]
    if unreadable:
        print(f"{len(unreadable)} images could not be read/encoded; they will be copied to 'unreadable'.")

    print("Classifying people vs scenery...")
    labels = classify_people_vs_scenery(embeddings, model, tokenizer, device)

    people_paths = [p for p, (lab, margin) in labels.items() if lab == "people" and margin >= args.uncertain_margin]
    scenery_paths = [p for p, (lab, margin) in labels.items() if lab == "scenery" and margin >= args.uncertain_margin]
    uncertain_paths = [p for p in labels if p not in people_paths and p not in scenery_paths]

    print(f"People: {len(people_paths)}, Scenery: {len(scenery_paths)}, Uncertain: {len(uncertain_paths)}")

    if args.region_mode:
        regions = sorted(set(region_of(p, args.input) for p in primaries))
    else:
        regions = [None]

    print("Clustering scenery photos per region..." if args.region_mode else "Clustering scenery photos...")
    cluster_map = {}
    for region in regions:
        if region is None:
            region_scenery = scenery_paths
        else:
            region_scenery = [p for p in scenery_paths if region_of(p, args.input) == region]
        cmap = cluster_scenery(region_scenery, embeddings, args.min_cluster_size)
        cluster_map.update(cmap)
        n_clusters = len(set(v for v in cmap.values() if v != -1))
        label = region if region else "(all)"
        print(f"  {label}: {len(region_scenery)} scenery photos -> {n_clusters} clusters")

    manifest = []
    errors = []
    transfer = shutil.move if args.move else shutil.copy2

    def base_dir(path):
        if not args.region_mode:
            return output_root
        return output_root / region_of(path, args.input)

    for p in tqdm(people_paths, desc="Copying people photos"):
        transfer_with_sidecars(p, base_dir(p) / "people", sidecars, transfer, manifest, "people", errors)

    for p in tqdm(uncertain_paths, desc="Copying uncertain photos"):
        transfer_with_sidecars(p, base_dir(p) / "uncertain", sidecars, transfer, manifest, "uncertain", errors)

    for p in tqdm(scenery_paths, desc="Copying scenery photos"):
        cluster_id = cluster_map[p]
        sub = "misc" if cluster_id == -1 else f"cluster_{cluster_id:03d}"
        transfer_with_sidecars(p, base_dir(p) / "scenery" / sub, sidecars, transfer, manifest, f"scenery/{sub}", errors)

    for p in tqdm(unreadable, desc="Copying unreadable photos"):
        transfer_with_sidecars(p, base_dir(p) / "unreadable", sidecars, transfer, manifest, "unreadable", errors)

    for f in tqdm(orphans, desc="Copying orphan RAW/XMP files"):
        region = region_of(f, args.input) if args.region_mode else None
        dest_dir = (output_root / region if region else output_root) / "raw_no_preview"
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            dest = unique_dest(dest_dir, f.name)
            transfer(str(f), str(dest))
            manifest.append({"src": str(f), "dest": str(dest), "category": "raw_no_preview"})
        except Exception as e:
            errors.append(f"{f}: {e}\n{traceback.format_exc()}")

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if errors:
        errors_path = output_root / "errors.log"
        with open(errors_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(errors))
        print(f"{len(errors)} files failed to copy; see {errors_path}")

    print(f"Done. {len(manifest)} files written. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
