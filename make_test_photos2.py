"""Generate a synthetic region+RAW+XMP mimic to smoke-test region-mode and sidecar carrying."""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("test_photos2")


def gradient(c1, c2, size=(256, 256)):
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    for y in range(size[1]):
        t = y / size[1]
        color = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
        draw.line([(0, y), (size[0], y)], fill=color)
    return img


def face(bg, size=(256, 256)):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    draw.ellipse([68, 48, 188, 188], fill=(230, 200, 170))
    draw.ellipse([95, 95, 110, 112], fill=(30, 30, 30))
    draw.ellipse([146, 95, 161, 112], fill=(30, 30, 30))
    draw.arc([100, 130, 160, 165], start=20, end=160, fill=(120, 60, 60), width=4)
    return img


def sidecar(path, content=b"dummy raw bytes"):
    path.write_bytes(content)


regionA = OUT / "RegionA"
regionB = OUT / "RegionB"
regionA.mkdir(parents=True, exist_ok=True)
regionB.mkdir(parents=True, exist_ok=True)

gradient((90, 150, 230), (200, 220, 255)).save(regionA / "IMG001.jpg")
sidecar(regionA / "IMG001.arw")
sidecar(regionA / "IMG001.xmp", b"<xmp/>")

gradient((95, 155, 225), (195, 218, 250)).save(regionA / "IMG002.jpg")
sidecar(regionA / "IMG002.arw")

gradient((100, 160, 220), (190, 215, 248)).save(regionA / "IMG003.jpg")
sidecar(regionA / "IMG003.arw")

face((240, 240, 240)).save(regionA / "IMG004.jpg")
sidecar(regionA / "IMG004.arw")

gradient((30, 90, 40), (100, 160, 60)).save(regionB / "IMG010.jpg")
sidecar(regionB / "IMG010.arw")

face((220, 220, 250)).save(regionB / "IMG011.jpg")
sidecar(regionB / "IMG011.arw")

sidecar(regionB / "IMG012.arw")  # orphan raw, no jpg pair

print("generated test_photos2 with", sum(1 for _ in OUT.rglob('*') if _.is_file()), "files")
