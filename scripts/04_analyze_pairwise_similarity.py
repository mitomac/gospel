#!/usr/bin/env python3
"""
04_analyze_pairwise_similarity.py - Analyze pairwise similarity between gospel verses

Computes cosine similarity between all verse pairs across Luke, Matthew, and Mark
Generates similarity matrices and identifies top parallel passages
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple

def load_data():
    """Load embeddings and metadata."""
    embeddings = np.load("data/embeddings/embeddings.npy")

    with open("data/embeddings/metadata.json", "r") as f:
        metadata = json.load(f)

    return embeddings, metadata

def create_book_indices(metadata: List[Dict]) -> Dict[str, List[int]]:
    """Create indices for each gospel book."""
    indices = {"Matthew": [], "Mark": [], "Luke": []}

    for idx, verse in enumerate(metadata):
        book = verse["book"]
        if book in indices:
            indices[book].append(idx)

    return indices

def compute_pairwise_similarities(
    embeddings: np.ndarray,
    indices_a: List[int],
    indices_b: List[int]
) -> np.ndarray:
    """Compute cosine similarity between two sets of verses."""
    embeddings_a = embeddings[indices_a]
    embeddings_b = embeddings[indices_b]

    similarities = cosine_similarity(embeddings_a, embeddings_b)
    return similarities

def find_top_matches(
    similarities: np.ndarray,
    metadata_a: List[Dict],
    metadata_b: List[Dict],
    top_k: int = 10
) -> List[Dict]:
    """Find top k most similar verse pairs."""
    # Get flattened indices of top k similarities
    flat_indices = np.argsort(similarities.flatten())[::-1][:top_k]

    matches = []
    for flat_idx in flat_indices:
        row = flat_idx // similarities.shape[1]
        col = flat_idx % similarities.shape[1]

        score = similarities[row, col]
        verse_a = metadata_a[row]
        verse_b = metadata_b[col]

        matches.append({
            "similarity": float(score),
            "verse_a": {
                "ref": f"{verse_a['book']} {verse_a['chapter']}:{verse_a['verse']}",
                "text": verse_a['text']
            },
            "verse_b": {
                "ref": f"{verse_b['book']} {verse_b['chapter']}:{verse_b['verse']}",
                "text": verse_b['text']
            }
        })

    return matches

def create_similarity_heatmap_data(
    similarities: np.ndarray,
    book_a: str,
    book_b: str,
    metadata_a: List[Dict],
    metadata_b: List[Dict]
) -> Dict:
    """Create summary statistics for similarity matrix."""
    summary = {
        "book_pair": f"{book_a} vs {book_b}",
        "num_verses_a": len(metadata_a),
        "num_verses_b": len(metadata_b),
        "mean_similarity": float(np.mean(similarities)),
        "max_similarity": float(np.max(similarities)),
        "min_similarity": float(np.min(similarities)),
        "high_similarity_pairs": int(np.sum(similarities > 0.8)),  # >80% similar
        "medium_similarity_pairs": int(np.sum((similarities > 0.5) & (similarities <= 0.8))),
        "low_similarity_pairs": int(np.sum(similarities <= 0.5))
    }

    return summary

def main():
    """Analyze pairwise similarity between synoptic gospels."""
    print("Gospel Pairwise Similarity Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading embeddings and metadata...")
    embeddings, metadata = load_data()
    print(f"✓ Loaded {len(embeddings)} verse embeddings")

    # Create book indices
    book_indices = create_book_indices(metadata)
    for book, indices in book_indices.items():
        print(f"  {book}: {len(indices)} verses")

    # Analyze all gospel pairs
    gospel_pairs = [
        ("Matthew", "Mark"),
        ("Matthew", "Luke"),
        ("Mark", "Luke")
    ]

    output_dir = Path("output/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_top_matches = {}

    print("\n" + "=" * 60)
    print("Computing pairwise similarities...")
    print("=" * 60)

    for book_a, book_b in gospel_pairs:
        print(f"\n{book_a} vs {book_b}:")
        print("-" * 60)

        # Get indices and metadata for each book
        indices_a = book_indices[book_a]
        indices_b = book_indices[book_b]
        metadata_a = [metadata[i] for i in indices_a]
        metadata_b = [metadata[i] for i in indices_b]

        # Compute similarities
        print(f"  Computing {len(indices_a)} x {len(indices_b)} similarities...", end=" ")
        similarities = compute_pairwise_similarities(embeddings, indices_a, indices_b)
        print("✓")

        # Create summary statistics
        summary = create_similarity_heatmap_data(
            similarities, book_a, book_b, metadata_a, metadata_b
        )
        all_summaries.append(summary)

        print(f"  Mean similarity: {summary['mean_similarity']:.3f}")
        print(f"  Max similarity: {summary['max_similarity']:.3f}")
        print(f"  High similarity pairs (>0.8): {summary['high_similarity_pairs']}")

        # Find top matches
        print(f"  Finding top 20 most similar verse pairs...", end=" ")
        top_matches = find_top_matches(similarities, metadata_a, metadata_b, top_k=20)
        all_top_matches[f"{book_a}_vs_{book_b}"] = top_matches
        print("✓")

        # Save full similarity matrix
        matrix_file = output_dir / f"similarity_matrix_{book_a.lower()}_vs_{book_b.lower()}.npy"
        np.save(matrix_file, similarities)
        print(f"  Saved matrix to: {matrix_file.name}")

    # Save summary statistics
    summary_file = output_dir / "similarity_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✓ Saved summary statistics to: {summary_file}")

    # Save top matches
    matches_file = output_dir / "top_parallel_passages.json"
    with open(matches_file, "w", encoding="utf-8") as f:
        json.dump(all_top_matches, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved top matches to: {matches_file}")

    # Print sample results
    print("\n" + "=" * 60)
    print("Sample Top Matches (Matthew vs Mark):")
    print("=" * 60)
    for i, match in enumerate(all_top_matches["Matthew_vs_Mark"][:5], 1):
        print(f"\n{i}. Similarity: {match['similarity']:.3f}")
        print(f"   {match['verse_a']['ref']}: {match['verse_a']['text'][:80]}...")
        print(f"   {match['verse_b']['ref']}: {match['verse_b']['text'][:80]}...")

    print("\n" + "=" * 60)
    print("✓ Analysis complete!")
    print(f"\nResults saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
