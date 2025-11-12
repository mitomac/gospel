#!/usr/bin/env python3
"""
01_get_data.py - Download gospel texts from Bible API

Downloads Matthew, Mark, and Luke texts using bible-api.com (free, no auth required)
Saves raw data to data/raw/
"""

import json
import requests
import time
from pathlib import Path

# API endpoint
BASE_URL = "https://bible-api.com"

# Synoptic gospels with verse counts
GOSPELS = {
    "Matthew": {"book": "matthew", "chapters": 28},
    "Mark": {"book": "mark", "chapters": 16},
    "Luke": {"book": "luke", "chapters": 24}
}

def fetch_chapter(book: str, chapter: int) -> dict:
    """Fetch a single chapter from the API."""
    url = f"{BASE_URL}/{book}+{chapter}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching {book} {chapter}: {e}")
        return None

def main():
    """Download all synoptic gospel chapters."""
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading synoptic gospels from bible-api.com...")
    print("=" * 60)

    for gospel_name, gospel_info in GOSPELS.items():
        book = gospel_info["book"]
        num_chapters = gospel_info["chapters"]

        print(f"\nDownloading {gospel_name} ({num_chapters} chapters)...")

        gospel_data = {
            "book": gospel_name,
            "chapters": []
        }

        for chapter in range(1, num_chapters + 1):
            print(f"  Chapter {chapter}/{num_chapters}...", end=" ")

            chapter_data = fetch_chapter(book, chapter)
            if chapter_data:
                gospel_data["chapters"].append(chapter_data)
                print("✓")
            else:
                print("✗")

            # Be nice to the API
            time.sleep(0.5)

        # Save to file
        output_file = output_dir / f"{book}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(gospel_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved to {output_file}")

    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print(f"\nRaw data saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
