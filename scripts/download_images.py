#!/usr/bin/env python3
"""Automated image dataset downloader using Wikimedia Commons API.

Features:
- Downloads up to 50 high-quality, diverse images per class across 10 object categories.
- Uses diversified search terms covering multiple angles, materials, lighting, and backgrounds.
- Perceptual hashing (dHash) to eliminate identical and near-duplicate images.
- Strict image verification using PIL (size, corruption, aspect ratio, color mode).
- Retry logic with exponential backoff for network resilience and API rate limits.
- Preserves existing valid images and supports missing-only downloads.
- Generates and updates license attribution metadata in dataset/attribution.csv.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image

# Setup paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
ATTRIBUTION_PATH = DATASET_DIR / "attribution.csv"

API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "AIObjectRecognitionStudentBot/2.0 "
    "(https://github.com/ml-student/object-recognition; contact: student-ml@university.edu)"
)

# 10 Object Classes with diversified search terms
QUERY_MAP = {
    "water_bottle": [
        "water bottle",
        "plastic water bottle",
        "reusable water bottle",
        "sports water bottle",
        "stainless water bottle",
        "drinking bottle",
        "water bottle container",
        "water bottle beverage",
    ],
    "coffee_mug": [
        "coffee mug",
        "ceramic mug",
        "drinking mug",
        "coffee cup",
        "tea mug",
        "porcelain mug",
        "coffee cup mug",
        "ceramic cup",
        "white coffee mug",
        "mug beverage",
    ],
    "smartphone": [
        "smartphone",
        "mobile phone",
        "touchscreen phone",
        "cell phone device",
        "android smartphone",
        "cellular phone",
        "smart phone",
    ],
    "book": [
        "book object",
        "hardcover book",
        "open book",
        "paperback book",
        "reading book",
        "novel book",
        "closed book",
    ],
    "computer_mouse": [
        "computer mouse",
        "optical mouse",
        "wireless mouse",
        "PC mouse",
        "USB mouse",
        "computer mouse device",
        "cordless mouse",
        "trackball mouse",
    ],
    "keyboard": [
        "computer keyboard",
        "PC keyboard",
        "desktop keyboard",
        "mechanical keyboard",
        "QWERTY keyboard",
        "USB keyboard",
    ],
    "tv_remote": [
        "television remote control",
        "TV remote control",
        "remote control",
        "infrared remote control",
        "TV remote controller",
        "television remote",
    ],
    "shoe": [
        "shoe footwear",
        "sneaker shoe",
        "running shoe",
        "leather shoe",
        "athletic shoe",
        "walking shoe",
    ],
    "scissors": [
        "scissors",
        "pair of scissors",
        "cutting scissors",
        "scissors tool",
        "office scissors",
        "craft scissors",
        "steel scissors",
    ],
    "wallet": [
        "wallet",
        "leather wallet",
        "purse wallet",
        "money wallet",
        "bifold wallet",
        "pocket wallet",
        "billfold wallet",
        "men wallet",
        "women wallet",
        "coin purse wallet",
    ],
}


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) to detect duplicate/near-duplicate images."""
    # Resize to (hash_size + 1, hash_size) grayscale
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            diff.append(left > right)
    decimal_val = 0
    for bit in diff:
        decimal_val = (decimal_val << 1) | bit
    return decimal_val


def hamming_distance(h1: int, h2: int) -> int:
    """Return the bitwise Hamming distance between two 64-bit hashes."""
    return bin(h1 ^ h2).count("1")


def is_near_duplicate(candidate_hash: int, existing_hashes: set[int], threshold: int = 6) -> bool:
    """Check if candidate hash is within hamming distance threshold of any known hash."""
    for h in existing_hashes:
        if hamming_distance(candidate_hash, h) <= threshold:
            return True
    return False


def request_with_retry(url: str, max_retries: int = 4, backoff: float = 1.5) -> bytes:
    """Fetch URL with exponential backoff for rate limits and intermittent network drops."""
    delay = backoff
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(req, timeout=35) as response:
                return response.read()
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except (URLError, TimeoutError) as e:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")


def search_wikimedia(query: str, limit: int = 50) -> list[dict]:
    """Query Wikimedia Commons API for bitmap images matching search terms."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,  # File: namespace
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 800,
    }
    url = f"{API_ENDPOINT}?{urlencode(params)}"
    raw = request_with_retry(url)
    data = json.loads(raw.decode("utf-8"))
    pages = data.get("query", {}).get("pages", {})
    return sorted(pages.values(), key=lambda p: p.get("title", ""))


def strip_html(text: str) -> str:
    """Clean HTML tags from metadata fields."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", "", text).replace("\n", " ").strip()


def validate_image_bytes(raw_bytes: bytes) -> tuple[Image.Image | None, int | None, str | None]:
    """Validate image bytes with PIL, ensure resolution and color mode, and compute dHash."""
    if len(raw_bytes) < 8_000:
        return None, None, "File size too small (< 8KB)"
    try:
        # First verify integrity
        test_stream = io.BytesIO(raw_bytes)
        with Image.open(test_stream) as img:
            img.verify()

        # Reopen for actual processing
        stream = io.BytesIO(raw_bytes)
        img = Image.open(stream)
        w, h = img.size
        if w < 224 or h < 224:
            return None, None, f"Resolution too low ({w}x{h})"

        # Exclude extreme aspect ratios (banners or thin stripes)
        aspect = max(w / h, h / w)
        if aspect > 3.8:
            return None, None, f"Extreme aspect ratio ({aspect:.2f})"

        # Convert to RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")

        dhash = compute_dhash(img)
        return img, dhash, None
    except Exception as e:
        return None, None, f"Corrupted or invalid image: {e}"


def get_existing_hashes(class_dir: Path) -> tuple[list[Path], set[int]]:
    """Scan existing images in class folder, validate them, and compute hashes."""
    valid_files = []
    hashes = set()
    if not class_dir.exists():
        return valid_files, hashes

    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    if img.width >= 200 and img.height >= 200:
                        img_rgb = img.convert("RGB")
                        h = compute_dhash(img_rgb)
                        hashes.add(h)
                        valid_files.append(path)
            except Exception:
                # Corrupt file; ignore or skip
                pass
    return valid_files, hashes


def download_dataset(
    target_per_class: int = 50,
    overwrite: bool = False,
    progress_callback=None,
) -> dict:
    """Execute the full dataset download across all 10 categories."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing_attribution = {}
    if ATTRIBUTION_PATH.exists() and not overwrite:
        try:
            with ATTRIBUTION_PATH.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_attribution[(row.get("class", ""), row.get("filename", ""))] = row
        except Exception as e:
            print(f"[Warning] Could not parse existing attribution CSV: {e}")

    final_stats = {}
    new_attribution_rows = list(existing_attribution.values())

    print("=" * 70)
    print("  10-CLASS OBJECT RECOGNITION DATASET DOWNLOADER")
    print(f"  Target: {target_per_class} valid images per class (Total: {target_per_class * len(QUERY_MAP)})")
    print(f"  Perceptual duplicate detection: ENABLED (dHash)")
    print(f"  Target directory: {DATASET_DIR}")
    print("=" * 70)

    for class_idx, (class_name, queries) in enumerate(QUERY_MAP.items(), 1):
        class_dir = DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        if overwrite:
            for p in class_dir.iterdir():
                if p.is_file() and p.name != ".gitkeep":
                    p.unlink()

        valid_files, known_hashes = get_existing_hashes(class_dir)
        current_count = len(valid_files)
        needed = max(0, target_per_class - current_count)

        print(f"\n[{class_idx:02d}/10] {class_name.upper()}:")
        print(f"       Current Valid: {current_count} | Needed: {needed} | Target: {target_per_class}")

        if progress_callback:
            progress_callback(class_name, current_count, target_per_class, "Scanning")

        if needed <= 0:
            print(f"       -> Target already satisfied ({current_count}/{target_per_class}). Skipping.")
            final_stats[class_name] = {
                "initial": current_count,
                "downloaded": 0,
                "total": current_count,
                "target": target_per_class,
                "status": "Complete",
            }
            continue

        downloaded_in_session = 0
        seen_titles = set()

        for q_idx, query in enumerate(queries, 1):
            if current_count >= target_per_class:
                break
            print(f"       Searching query {q_idx}/{len(queries)}: '{query}'...")
            try:
                pages = search_wikimedia(query, limit=50)
            except Exception as e:
                print(f"       [Query Error] {e}")
                continue

            for page in pages:
                if current_count >= target_per_class:
                    break
                title = page.get("title", "")
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                image_infos = page.get("imageinfo", [])
                if not image_infos:
                    continue
                info = image_infos[0]
                mime = info.get("mime", "")
                if mime not in {"image/jpeg", "image/png", "image/jpg"}:
                    continue

                img_url = info.get("url")
                if not img_url:
                    continue

                try:
                    raw_content = request_with_retry(img_url, max_retries=3)
                    img_obj, dhash, err_msg = validate_image_bytes(raw_content)
                    if err_msg or img_obj is None or dhash is None:
                        continue

                    # Duplicate check
                    if is_near_duplicate(dhash, known_hashes, threshold=6):
                        continue

                    # Valid, unique image found! Save as JPG
                    filename = f"{class_name}_{current_count + 1:03d}.jpg"
                    dest_path = class_dir / filename
                    img_obj.save(dest_path, "JPEG", quality=92, optimize=True)

                    known_hashes.add(dhash)
                    current_count += 1
                    downloaded_in_session += 1

                    meta = info.get("extmetadata", {})
                    author = strip_html(meta.get("Artist", {}).get("value", "Unknown"))
                    license_name = strip_html(meta.get("LicenseShortName", {}).get("value", "Public Domain / CC"))
                    license_url = meta.get("LicenseUrl", {}).get("value", "")

                    attr_row = {
                        "class": class_name,
                        "filename": filename,
                        "commons_title": title,
                        "author": author[:100],
                        "license": license_name[:50],
                        "license_url": license_url,
                        "source_page": info.get("descriptionurl", ""),
                        "download_url": img_url,
                    }
                    new_attribution_rows.append(attr_row)

                    print(f"       + [{current_count}/{target_per_class}] Saved: {filename} (hash: {dhash:016x})")
                    if progress_callback:
                        progress_callback(class_name, current_count, target_per_class, f"Saved {filename}")

                    time.sleep(0.35)  # Respectful rate limiting
                except Exception as dl_err:
                    print(f"       [Skip] Failed {title[:40]}: {dl_err}")

        status = "Complete" if current_count >= target_per_class else f"Shortfall (-{target_per_class - current_count})"
        final_stats[class_name] = {
            "initial": len(valid_files),
            "downloaded": downloaded_in_session,
            "total": current_count,
            "target": target_per_class,
            "status": status,
        }

    # Write attribution CSV
    fieldnames = [
        "class",
        "filename",
        "commons_title",
        "author",
        "license",
        "license_url",
        "source_page",
        "download_url",
    ]
    with ATTRIBUTION_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_attribution_rows)

    print("\n" + "=" * 70)
    print("  DOWNLOAD SUMMARY REPORT")
    print("=" * 70)
    print(f"{'Category':<18} | {'Initial':<8} | {'Added':<6} | {'Total Valid':<12} | {'Target':<6} | {'Status'}")
    print("-" * 70)
    total_valid_all = 0
    for c_name, st in final_stats.items():
        total_valid_all += st["total"]
        print(
            f"{c_name:<18} | {st['initial']:<8} | {st['downloaded']:<6} | "
            f"{st['total']:<12} | {st['target']:<6} | {st['status']}"
        )
    print("-" * 70)
    print(f"TOTAL VALID IMAGES ACROSS ALL CATEGORIES: {total_valid_all} / {target_per_class * len(QUERY_MAP)}")
    print(f"License attribution saved to: {ATTRIBUTION_PATH}")
    print("=" * 70)

    return final_stats


def main():
    parser = argparse.ArgumentParser(description="Download object images from Wikimedia Commons with deduplication.")
    parser.add_argument("--per-class", type=int, default=50, help="Target valid images per class (default: 50)")
    parser.add_argument("--overwrite", action="store_true", help="Clear existing images before downloading")
    parser.add_argument("--missing-only", action="store_true", help="Download only missing images to hit target")
    args = parser.parse_args()

    download_dataset(
        target_per_class=args.per_class,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
