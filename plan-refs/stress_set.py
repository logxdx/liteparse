#!/usr/bin/env python3
"""Build an OCR stress set from the ocr-bench corpus.

Each variant keeps the ground truth of its source page (the text is unchanged,
only the image is). Multilingual pages are rendered from fixed sentences.
Usage: stress_set.py <corpus_dir> <out_dir> [pages...]
Writes <out_dir>/img/<variant>__<page>.png and <out_dir>/gt/<variant>__<page>.txt
plus <out_dir>/variants.json (variant -> group, lang).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

corpus, out = Path(sys.argv[1]), Path(sys.argv[2])
pages = sys.argv[3:] or ["p002", "p011", "p045", "p090", "p130", "p164"]
(out / "img").mkdir(parents=True, exist_ok=True)
(out / "gt").mkdir(parents=True, exist_ok=True)


def rotate_quarter(g, k):
    return np.rot90(g, k=k).copy()


def tilt(g, deg):
    h, w = g.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(g, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)


def small(g):
    return cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def tiny(g):
    return cv2.resize(g, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)


def inverted(g):
    return 255 - g


def noisy(g):
    rng = np.random.default_rng(0)
    b = cv2.GaussianBlur(g, (0, 0), 1.2).astype(np.float32)
    b += rng.normal(0, 18, g.shape)
    return np.clip(b, 0, 255).astype(np.uint8)


def photo(g):
    """Uneven lighting plus a 2 degree tilt: a phone snapshot of a page."""
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 0.62 + 0.38 * (xx / w * 0.6 + yy / h * 0.4)
    return tilt(np.clip(g * shade, 0, 255).astype(np.uint8), 2.0)


VARIANTS = {
    "upright": ("baseline", lambda g: g),
    "rot90": ("rotation", lambda g: rotate_quarter(g, 1)),
    "rot180": ("rotation", lambda g: rotate_quarter(g, 2)),
    "rot270": ("rotation", lambda g: rotate_quarter(g, 3)),
    "tilt3": ("tilt", lambda g: tilt(g, 3)),
    "tiltm10": ("tilt", lambda g: tilt(g, -10)),
    "tilt25": ("tilt", lambda g: tilt(g, 25)),
    "half_res": ("small", small),
    "tiny_res": ("small", tiny),
    "inverted": ("dark", inverted),
    "noisy": ("degraded", noisy),
    "photo": ("degraded", photo),
}
meta = {}
for v, (group, fn) in VARIANTS.items():
    for p in pages:
        g = cv2.imread(str(corpus / "img" / f"{p}.png"), cv2.IMREAD_GRAYSCALE)
        name = f"{v}__{p}"
        cv2.imwrite(str(out / "img" / f"{name}.png"), fn(g))
        (out / "gt" / f"{name}.txt").write_text((corpus / "gt" / f"{p}.txt").read_text())
        meta[name] = {"group": group, "lang": "eng"}

# Multilingual: (name, tesseract lang, font, text lines)
FONTS = "/usr/share/fonts"


def font(pattern, size):
    hits = sorted(Path(FONTS).rglob(pattern))
    if not hits:
        raise SystemExit(f"font not found: {pattern}")
    return ImageFont.truetype(str(hits[0]), size)


SCRIPTS = [
    ("cyrillic", "rus", "NotoSans-Regular.ttf", [
        "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "Документ содержит несколько строк текста для проверки.",
        "Распознавание текста должно работать на разных языках.",
        "Таблица результатов приведена в приложении к отчёту."]),
    ("greek", "ell", "NotoSans-Regular.ttf", [
        "Η γρήγορη καφέ αλεπού πηδάει πάνω από το τεμπέλικο σκυλί.",
        "Το έγγραφο περιέχει αρκετές γραμμές κειμένου για έλεγχο.",
        "Η αναγνώριση κειμένου πρέπει να λειτουργεί σε πολλές γλώσσες.",
        "Ο πίνακας αποτελεσμάτων βρίσκεται στο παράρτημα της έκθεσης."]),
    ("devanagari", "hin", "NotoSansDevanagari-Regular.ttf", [
        "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
        "इस दस्तावेज़ में जाँच के लिए कई पंक्तियाँ हैं।",
        "पाठ पहचान को कई भाषाओं में काम करना चाहिए।",
        "परिणामों की तालिका रिपोर्ट के परिशिष्ट में है।"]),
    ("chinese", "chi_sim", "NotoSansCJK-Regular.ttc", [
        "敏捷的棕色狐狸跳过了懒惰的狗。",
        "这份文件包含多行文字用于检查识别效果。",
        "文字识别应当适用于多种不同的语言。",
        "结果表格见报告的附录部分。"]),
    ("arabic", "ara", "NotoSansArabic-Regular.ttf", [
        "الثعلب البني السريع يقفز فوق الكلب الكسول.",
        "تحتوي هذه الوثيقة على عدة أسطر من النص للاختبار.",
        "يجب أن يعمل التعرف على النص بلغات مختلفة.",
        "جدول النتائج موجود في ملحق التقرير."]),
]
for name, lang, ff, lines in SCRIPTS:
    f = font(ff, 34)
    img = Image.new("L", (1700, 90 + 70 * len(lines)), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((60, 50 + 70 * i), line, font=f, fill=0, direction="rtl" if lang == "ara" else None)
    key = f"script_{name}__page"
    img.save(out / "img" / f"{key}.png")
    (out / "gt" / f"{key}.txt").write_text("\n".join(lines) + "\n")
    meta[key] = {"group": "multilingual", "lang": lang}

(out / "variants.json").write_text(json.dumps(meta, indent=1))
print(len(meta), "images")
