"""
Local web app to browse and fix the organized photo library.
PC:    double-click start_photo_app.bat  ->  http://127.0.0.1:8765
Phone: double-click start_photo_app_lan.bat, open http://<PC-LAN-IP>:8765 on same WiFi.
"""
import argparse
import re
import shutil
import socket
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageOps
from pydantic import BaseModel

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff"}
SIDECAR_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".xmp"}
THUMB_SIZE = 360

app = FastAPI()
ROOT = Path(".")
THUMB_DIR = Path(".")
APP_DIR = Path(__file__).parent


def safe_path(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if not p.is_relative_to(ROOT.resolve()):
        raise HTTPException(status_code=400, detail="bad path")
    return p


def count_images(d: Path) -> int:
    return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


@app.get("/", response_class=HTMLResponse)
def index():
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/tree")
def tree():
    regions = []
    for r in sorted(ROOT.iterdir()):
        if not r.is_dir() or r.name.startswith("."):
            continue
        cats = []
        for sub in ("people", "uncertain", "unreadable", "raw_no_preview"):
            d = r / sub
            if d.is_dir():
                cats.append({"name": sub, "rel": str(d.relative_to(ROOT)), "count": count_images(d)})
        scenery = r / "scenery"
        if scenery.is_dir():
            for c in sorted(scenery.iterdir()):
                if c.is_dir():
                    cats.append({"name": f"scenery/{c.name}", "rel": str(c.relative_to(ROOT)), "count": count_images(c)})
        if cats:
            regions.append({"region": r.name, "cats": cats})
    return regions


@app.get("/api/photos")
def photos(rel: str):
    d = safe_path(rel)
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="folder not found")
    out = []
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            out.append({"rel": str(f.relative_to(ROOT)), "name": f.name})
    return out


@app.get("/thumb")
def thumb(rel: str):
    src = safe_path(rel)
    if not src.is_file():
        raise HTTPException(status_code=404, detail="not found")
    cache_file = THUMB_DIR / Path(rel).with_suffix(".jpg")
    if cache_file.exists() and cache_file.stat().st_mtime >= src.stat().st_mtime:
        return FileResponse(cache_file, media_type="image/jpeg")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src)
        img.draft("RGB", (THUMB_SIZE * 2, THUMB_SIZE * 2))
        img = ImageOps.exif_transpose(img)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        img.convert("RGB").save(cache_file, "JPEG", quality=82)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"thumbnail failed: {e}")
    return FileResponse(cache_file, media_type="image/jpeg")


@app.get("/full")
def full(rel: str):
    f = safe_path(rel)
    if not f.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(f)


class MoveReq(BaseModel):
    items: list[str]
    dest: str


@app.post("/api/move")
def move(req: MoveReq):
    dest_dir = safe_path(req.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for rel in req.items:
        src = safe_path(rel)
        if not src.is_file():
            continue
        group = [src] + [s for s in src.parent.iterdir()
                         if s.is_file() and s.stem.lower() == src.stem.lower() and s.suffix.lower() in SIDECAR_EXTS]
        for f in group:
            target = dest_dir / f.name
            i = 1
            while target.exists():
                target = dest_dir / f"{f.stem}_{i}{f.suffix}"
                i += 1
            shutil.move(str(f), str(target))
        moved += 1
    return {"moved": moved}


class RenameReq(BaseModel):
    rel: str
    new_name: str


@app.post("/api/rename")
def rename(req: RenameReq):
    d = safe_path(req.rel)
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="folder not found")
    name = re.sub(r'[\\/:*?"<>|]', "", req.new_name).strip()
    if not name:
        raise HTTPException(status_code=400, detail="bad name")
    new_dir = d.with_name(name)
    if new_dir.exists():
        raise HTTPException(status_code=409, detail="name already exists")
    d.rename(new_dir)
    return {"rel": str(new_dir.relative_to(ROOT))}


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\暑假旅程_整理")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ROOT = Path(args.root)
    THUMB_DIR = ROOT / ".thumbs"
    if not ROOT.is_dir():
        raise SystemExit(f"root not found: {ROOT}")
    print(f"Serving {ROOT}")
    print(f"  PC:    http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  Phone (same WiFi): http://{lan_ip()}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
