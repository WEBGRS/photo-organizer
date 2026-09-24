# Regenerate README screenshots with a demo library of Picsum (Unsplash License) photos
import subprocess, sys, tempfile, time, pathlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
PORT = 8799
LIBRARY = {
    "Norway/scenery/cluster_001_coast": [10, 11, 12, 13, 14, 16, 37, 147, 179, 124],
    "Norway/scenery/cluster_002_snow_mountain": [29, 79, 166, 66, 191, 118],
    "Norway/scenery/cluster_003_forest": [15, 17, 28, 18, 83, 190, 81],
    "California/scenery/cluster_001_city": [43, 57, 61, 84, 122, 164, 88],
    "California/scenery/cluster_002_desert": [184, 196, 185, 110],
}

# Build demo library
lib = pathlib.Path(tempfile.mkdtemp()) / "library"
jobs = []
for folder, ids in LIBRARY.items():
    (lib / folder).mkdir(parents=True)
    jobs += [(lib / folder / f"IMG_{i:04d}.jpg", i) for i in ids]
def fetch(job):
    path, i = job
    path.write_bytes(urllib.request.urlopen(f"https://picsum.photos/id/{i}/1200/800", timeout=60).read())
with ThreadPoolExecutor(12) as ex:
    list(ex.map(fetch, jobs))

srv = subprocess.Popen([sys.executable, str(ROOT / "app.py"), "--root", str(lib), "--port", str(PORT)])
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/tree"); break
        except OSError:
            time.sleep(0.5)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--lang=en-US"])
        page = b.new_page(viewport={"width": 1000, "height": 560}, locale="en-US")
        page.goto(f"http://127.0.0.1:{PORT}/")
        page.wait_for_selector(".cell img")
        page.select_option("#regionSel", label="Norway")
        page.wait_for_function("document.getElementById('count').textContent.startsWith('10')")
        page.wait_for_function("[...document.querySelectorAll('.cell img')].every(i => i.complete && i.naturalWidth)", timeout=60000)
        page.screenshot(path=str(OUT / "browse.png"))

        # Select a few photos and open the move dialog
        page.click("#selBtn")
        for n in (1, 4, 6):
            page.click(f".cell >> nth={n}")
        page.click("#moveBtn")
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "move.png"))
        b.close()
finally:
    srv.terminate()

# Shrink PNGs
from PIL import Image
for f in OUT.glob("*.png"):
    Image.open(f).convert("RGB").save(f.with_suffix(".jpg"), quality=85, optimize=True)
    f.unlink()
print("saved to", OUT)
