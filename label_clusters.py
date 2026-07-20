"""
Auto-name scenery cluster folders using cached CLIP embeddings.
cluster_003 -> cluster_003_雪山  (folder rename; manifest.json updated to match)

Usage:
    python label_clusters.py --output "D:\\暑假旅程_整理"
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import open_clip
import torch

from organize_photos import cache_key

# (CLIP English prompt, Chinese folder label)
LABELS = [
    ("snow-capped mountains and glaciers", "雪山"),
    ("a blue alpine lake", "湖泊"),
    ("a grassland meadow with yaks or horses", "草原牧场"),
    ("an old town street with lanterns and wooden shops", "古镇街巷"),
    ("traditional Chinese temple or pagoda architecture", "古建寺庙"),
    ("a modern city skyline with skyscrapers", "城市建筑"),
    ("a city street at night with neon lights", "城市夜景"),
    ("tropical rainforest plants and trees", "雨林植物"),
    ("colorful flowers in a garden", "花卉"),
    ("dishes of food on a table", "美食"),
    ("sky and dramatic clouds at sunset", "天空云彩"),
    ("a river, waterfall or stream", "河流瀑布"),
    ("a bridge over a river", "桥梁"),
    ("the interior of a cafe, hotel or museum", "室内"),
    ("animals or birds", "动物"),
    ("a winding road or hiking trail in the mountains", "山间道路"),
    ("a panoramic view of a valley from a viewpoint", "山谷全景"),
    ("a busy street market with crowds", "市集街景"),
    ("terraced fields or farmland", "田园梯田"),
    ("a peninsula, riverbend or aerial landscape view", "江湾俯瞰"),
]


def main():
    parser = argparse.ArgumentParser(description="Auto-name scenery cluster folders with CLIP.")
    parser.add_argument("--output", default=r"D:\暑假旅程_整理", help="Organized output root.")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    root = Path(args.output)
    manifest_path = root / "manifest.json"
    cache_path = root / ".embeddings_cache.pkl"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    # dest jpg -> embedding via original src file stat
    dest_emb = {}
    for entry in manifest:
        dest = Path(entry["dest"])
        if dest.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".heic"}:
            continue
        src = Path(entry["src"])
        if not src.exists():
            continue
        key = cache_key(src)
        if key in cache:
            dest_emb[dest.resolve()] = cache[key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    prompts = [f"a photo of {p}" for p, _ in LABELS]
    with torch.no_grad():
        text_feats = model.encode_text(tokenizer(prompts).to(device))
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().numpy()

    renames = []
    for region in sorted(root.iterdir()):
        scenery = region / "scenery"
        if not scenery.is_dir():
            continue
        for cluster in sorted(scenery.iterdir()):
            if not cluster.is_dir() or not re.fullmatch(r"cluster_\d{3}", cluster.name):
                continue
            embs = [dest_emb[p.resolve()] for p in cluster.iterdir() if p.resolve() in dest_emb]
            if not embs:
                print(f"[skip] {cluster}: no cached embeddings")
                continue
            centroid = np.mean(np.stack(embs), axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            scores = centroid @ text_feats.T
            label = LABELS[int(np.argmax(scores))][1]
            new_dir = cluster.with_name(f"{cluster.name}_{label}")
            cluster.rename(new_dir)
            renames.append((str(cluster), str(new_dir)))
            print(f"{region.name}\\{cluster.name} -> {new_dir.name}  (score {scores.max():.3f}, {len(embs)} photos)")

    # Update manifest dest paths for renamed folders
    for old, new in renames:
        for entry in manifest:
            if entry["dest"].startswith(old + "\\") or entry["dest"] == old:
                entry["dest"] = new + entry["dest"][len(old):]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Renamed {len(renames)} cluster folders; manifest updated.")


if __name__ == "__main__":
    main()
