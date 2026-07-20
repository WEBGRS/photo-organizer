"""Generate synthetic test images to smoke-test organize_photos.py."""
import random
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("test_photos")
OUT.mkdir(exist_ok=True)


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


random.seed(0)
for i in range(6):
    gradient((90 + i * 5, 150, 230 - i * 5), (200 + i * 5, 220, 255)).save(OUT / f"sky_{i:02d}.jpg")
for i in range(6):
    gradient((30, 90 + i * 5, 40), (100, 160 - i * 3, 60)).save(OUT / f"forest_{i:02d}.jpg")
for i in range(5):
    bg = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
    face(bg).save(OUT / f"person_{i:02d}.jpg")

print("Generated", len(list(OUT.glob('*.jpg'))), "test images in", OUT.resolve())
