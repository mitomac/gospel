#!/usr/bin/env python3
"""
02_process_data.py - Parse and structure verse data

Reads raw JSON from data/raw/ and creates structured verse objects
Saves to data/processed/verses.json
"""

import json
import re
from pathlib import Path
from typing import List, Dict

def parse_verses(chapter_data: dict, book: str) -> List[Dict]:
    """
    Parse verses from a chapter's data.

    The API returns a 'verses' array with structured verse objects.
    """
    verses = []

    # Get the verses array
    verse_array = chapter_data.get("verses", [])

    for verse_obj in verse_array:
        chapter_num = verse_obj.get("chapter")
        verse_num = verse_obj.get("verse")
        verse_text = verse_obj.get("text", "").strip()

        verse_id = f"{book.lower()}_{chapter_num}_{verse_num}"

        verses.append({
            "id": verse_id,
            "book": book,
            "chapter": chapter_num,
            "verse": verse_num,
            "text": verse_text,
            "translation": "WEB"  # bible-api.com uses World English Bible
        })

    return verses

def main():
    """Process all downloaded gospel data."""
    raw_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Processing gospel data...")
    print("=" * 60)

    all_verses = []
    stats = {}

    # Process each gospel
    for gospel_file in sorted(raw_dir.glob("*.json")):
        book_name = gospel_file.stem.capitalize()
        print(f"\nProcessing {book_name}...")

        with open(gospel_file, "r", encoding="utf-8") as f:
            gospel_data = json.load(f)

        book_verses = []
        for chapter_data in gospel_data["chapters"]:
            chapter_verses = parse_verses(chapter_data, book_name)
            book_verses.extend(chapter_verses)
            all_verses.extend(chapter_verses)

        stats[book_name] = len(book_verses)
        print(f"  {len(book_verses)} verses processed")

    # Save all verses
    output_file = output_dir / "verses.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_verses, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print(f"\nTotal verses: {len(all_verses)}")
    for book, count in stats.items():
        print(f"  {book}: {count} verses")
    print(f"\nProcessed data saved to: {output_file.absolute()}")

if __name__ == "__main__":
    main()
