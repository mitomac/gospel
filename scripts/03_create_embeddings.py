#!/usr/bin/env python3
"""
03_create_embeddings.py - Generate sentence embeddings for gospel verses

Uses paraphrase-multilingual-mpnet-base-v2 to create 768-dimensional embeddings
Saves embeddings to data/embeddings/
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict

def load_verses(verses_file: Path) -> List[Dict]:
    """Load processed verses from JSON file."""
    with open(verses_file, "r", encoding="utf-8") as f:
        return json.load(f)

def create_embeddings(verses: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """Create embeddings for all verses."""
    # Extract verse texts
    texts = [verse["text"] for verse in verses]

    print(f"Creating embeddings for {len(texts)} verses...")
    print("This may take a minute on first run (model download)...")

    # Generate embeddings in batches for efficiency
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    return embeddings

def main():
    """Generate embeddings for all gospel verses."""
    print("Gospel Verse Embeddings Generator")
    print("=" * 60)

    # Load processed verses
    verses_file = Path("data/processed/verses.json")
    if not verses_file.exists():
        print(f"Error: {verses_file} not found. Run 02_process_data.py first.")
        return

    verses = load_verses(verses_file)
    print(f"\n✓ Loaded {len(verses)} verses")

    # Load sentence transformer model
    print("\nLoading model: paraphrase-multilingual-mpnet-base-v2...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("✓ Model loaded")

    # Create embeddings
    print()
    embeddings = create_embeddings(verses, model)
    print(f"✓ Created embeddings: shape {embeddings.shape}")

    # Save embeddings and metadata
    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings as numpy array
    embeddings_file = output_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"\n✓ Saved embeddings to: {embeddings_file}")

    # Save verse metadata (for matching embeddings to verses)
    metadata = [
        {
            "id": v["id"],
            "book": v["book"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "text": v["text"]
        }
        for v in verses
    ]
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata to: {metadata_file}")

    print("\n" + "=" * 60)
    print("✓ Embedding generation complete!")

if __name__ == "__main__":
    main()
