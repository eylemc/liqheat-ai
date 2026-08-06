from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from supabase import create_client

BUCKET = "liq-logging-archive"
PAGE_SIZE = 1000

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SECRET_KEY")

if not url or not key:
    raise RuntimeError("SUPABASE_URL veya SUPABASE_SECRET_KEY eksik")

sb = create_client(url, key)
bucket = sb.storage.from_(BUCKET)


def get_value(item: Any, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def list_path(path: str = ""):
    offset = 0
    all_items = []

    while True:
        items = bucket.list(
            path=path,
            options={
                "limit": PAGE_SIZE,
                "offset": offset,
                "sortBy": {
                    "column": "name",
                    "order": "asc",
                },
            },
        )

        if not items:
            break

        all_items.extend(items)

        if len(items) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return all_items


files = []
folders_seen = set()


def walk(path: str = ""):
    items = list_path(path)

    for item in items:
        name = get_value(item, "name")
        if not name:
            continue

        full_path = f"{path}/{name}".strip("/")
        metadata = get_value(item, "metadata") or {}

        item_id = get_value(item, "id")
        size = metadata.get("size") if isinstance(metadata, dict) else None

        # Storage klasörleri çoğunlukla id/metadata içermeden döner.
        is_folder = item_id is None and size is None

        if is_folder:
            if full_path not in folders_seen:
                folders_seen.add(full_path)
                walk(full_path)
        else:
            files.append(
                {
                    "path": full_path,
                    "size": int(size or 0),
                    "created_at": get_value(item, "created_at"),
                    "updated_at": get_value(item, "updated_at"),
                }
            )


print(f"Bucket taranıyor: {BUCKET}")
walk()

total_size = sum(item["size"] for item in files)
extensions = Counter(
    os.path.splitext(item["path"])[1].lower() or "(uzantısız)"
    for item in files
)

print("\n=== ARCHIVE SUMMARY ===")
print("Klasör sayısı:", len(folders_seen))
print("Dosya sayısı:", len(files))
print("Toplam boyut:", f"{total_size / 1024 / 1024:.2f} MB")
print("Toplam boyut:", f"{total_size / 1024 / 1024 / 1024:.3f} GB")

print("\n=== EXTENSIONS ===")
for extension, count in extensions.most_common():
    print(f"{extension}: {count}")

if files:
    print("\n=== FIRST 10 FILES ===")
    for item in files[:10]:
        print(
            f'{item["size"]:>12} bytes  '
            f'{item["created_at"] or "-":<30}  '
            f'{item["path"]}'
        )

    print("\n=== LAST 10 FILES ===")
    for item in files[-10:]:
        print(
            f'{item["size"]:>12} bytes  '
            f'{item["created_at"] or "-":<30}  '
            f'{item["path"]}'
        )

    dated = [
        item for item in files
        if item["created_at"]
    ]

    if dated:
        dated.sort(key=lambda item: item["created_at"])
        print("\n=== STORAGE DATE RANGE ===")
        print("İlk oluşturulan:", dated[0]["created_at"], dated[0]["path"])
        print("Son oluşturulan:", dated[-1]["created_at"], dated[-1]["path"])
