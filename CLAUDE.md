# Gospel Synopsis Generator

A computational tool for creating gospel synopses (side-by-side comparisons of parallel passages in Matthew, Mark, and Luke) using RAG architecture: embeddings for retrieval + LLM for analysis.

## Project Goal

Build a tool that:
1. Finds parallel passages across synoptic gospels using semantic similarity
2. Analyzes differences (exact matches, paraphrases, unique material)
3. Generates color-coded synopsis tables

**Why**: Students manually creating synopses takes 30-60 min per passage. Conversational AI (ChatGPT) doesn't help because it lacks systematic retrieval. This tool uses proper RAG architecture: embeddings → retrieval → LLM analysis → structured output.

## Workflow: Numbered Scripts

This project uses incrementally numbered scripts for each pipeline step:

```
01_get_data.py          # Download gospel texts via API
02_process_data.py      # Parse and structure verse data
03_create_embeddings.py # Generate sentence embeddings
04_build_vectordb.py    # Create ChromaDB collection
05_find_parallels.py    # Vector search for similar verses
06_analyze_with_llm.py  # Claude analyzes candidates
07_generate_synopsis.py # Create formatted output
...
33_heatmap_analysis.py  # Advanced visualizations
```

**Numbering convention**: Two digits, increments of 1, descriptive names. Makes workflow order clear and easy to follow.

## Data Sources

### Gospel Statistics
- **Matthew**: 1,071 verses, ~18,345 words
- **Mark**: 678 verses, ~11,304 words
- **Luke**: 1,151 verses, ~19,482 words
- **Total**: ~2,900 verses, ~49,000 words

### API Options (Free, No Auth Required)

**1. bible-api.com** (Recommended for English)
```python
import requests
response = requests.get("https://bible-api.com/matthew+1")
data = response.json()
# Returns: verse text, reference, translation
```

**2. bolls.life/api** (Supports Greek + English)
```python
# Greek text (SBL GNT)
url = "https://bolls.life/get-paralel-verses/SBLGNT/40/1/"  # Matthew 1
# English (multiple translations)
url = "https://bolls.life/get-paralel-verses/kjv/40/1/"
```

**3. API.Bible** (Requires free API key)
```bash
curl -H "api-key: YOUR_KEY" "https://api.scripture.api.bible/v1/bibles/de4e12af7f28f599-02/verses/MAT.1.1"
```

**4. GitHub Repos** (Static JSON files)
- https://github.com/scrollmapper/bible_databases
- https://github.com/thiagobodruk/bible (JSON, multiple translations)

### Greek Text Sources

**SBL Greek New Testament** (Recommended)
- Free, scholarly standard
- Available via bolls.life API
- Public domain

**Alternative**: Nestle-Aland via academic APIs

## Sentence Transformers

### Model Selection

**Primary: `paraphrase-multilingual-mpnet-base-v2`**
- 768-dimensional embeddings
- Handles Greek + English in same embedding space
- Best quality for gospel comparison

**Alternative: `paraphrase-multilingual-MiniLM-L12-v2`**
- 384-dimensional (faster, smaller)
- Good for prototyping

**Why multilingual works**: Model trained on parallel Bible translations across 50+ languages, learns that Greek "ἐκτείνας τὴν χεῖρα" ≈ English "stretched out his hand"

### Installation & Usage

```python
pip install sentence-transformers

from sentence_transformers import SentenceTransformer

# Load model (downloads ~400MB first time)
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Embed verses
greek_text = "ἐκτείνας τὴν χεῖρα ἥψατο αὐτοῦ"
english_text = "he stretched out his hand and touched him"

greek_embedding = model.encode(greek_text)      # shape: (768,)
english_embedding = model.encode(english_text)  # shape: (768,)

# Cosine similarity (these should be very close!)
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([greek_embedding], [english_embedding])[0][0]
# Expected: >0.85 for parallel translations
```

## Vector Database

**ChromaDB** (Recommended for this project)
```python
pip install chromadb

import chromadb
client = chromadb.Client()

collection = client.create_collection(
    name="gospel_verses",
    metadata={"description": "Synoptic gospel verses with Greek + English"}
)

# Add verse with embedding
collection.add(
    ids=["matt_1_1"],
    embeddings=[embedding.tolist()],
    metadatas=[{
        "book": "Matthew",
        "chapter": 1,
        "verse": 1,
        "greek": "Βίβλος γενέσεως Ἰησοῦ Χριστοῦ...",
        "english": "The book of the genealogy of Jesus Christ...",
        "translation": "ESV"
    }]
)

# Query for similar verses
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5,
    where={"book": {"$ne": "Matthew"}}  # Exclude same book
)
```

## LLM Analysis

**Claude Sonnet 4** for analyzing retrieved candidates:

```python
pip install anthropic

from anthropic import Anthropic
client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": f"""Compare these gospel passages:

Matthew 8:3: "ἐκτείνας τὴν χεῖρα ἥψατο αὐτοῦ" / "he stretched out his hand and touched him"

Candidates from vector search:
1. Mark 1:41: "ἐκτείνας τὴν χεῖρα αὐτοῦ ἥψατο" / "he stretched out his hand and touched him"
2. Luke 5:13: "ἐκτείνας τὴν χεῖρα ἥψατο αὐτοῦ" / "he stretched out his hand and touched him"

For each candidate:
- Similarity score (0-100)
- Exact word matches (for green highlighting)
- Semantic matches (for yellow highlighting)
- Unique material in each version

Output as JSON."""
    }]
)
```

## Technical Architecture

```
01_get_data.py
   ↓ (fetches gospel texts)

02_process_data.py
   ↓ (structures as verse objects)

03_create_embeddings.py
   ↓ (sentence transformer → 768-dim vectors)

04_build_vectordb.py
   ↓ (ChromaDB collection)

05_find_parallels.py
   ↓ (cosine similarity search, top-k candidates)

06_analyze_with_llm.py
   ↓ (Claude compares Greek + English, outputs JSON)

07_generate_synopsis.py
   ↓ (color-coded HTML/PDF table)
```

## Data Structure

Each verse stored as:
```python
{
    "id": "matt_8_3",
    "book": "Matthew",
    "chapter": 8,
    "verse": 3,
    "greek": "ἐκτείνας τὴν χεῖρα ἥψατο αὐτοῦ λέγων...",
    "english": "And Jesus stretched out his hand and touched him...",
    "translation": "ESV",
    "embedding": [0.023, -0.145, 0.892, ...]  # 768 dimensions
}
```

## Output Format

### Color Coding Scheme
- **Green**: High similarity (>80%) - exact matches
- **Yellow**: Medium similarity (40-80%) - same meaning, different words
- **Pink**: Low similarity (<40%) - present but quite different
- **Gray**: Unique to one gospel

### Synopsis Table Structure
```
Matthew 8:2-4     | Mark 1:40-45      | Luke 5:12-16
------------------|-------------------|------------------
καὶ ἰδοὺ         | καὶ ἔρχεται      | καὶ ἐγένετο
(and behold)      | (and there comes) | (and it happened)
                  |                   |
λεπρὸς           | λεπρὸς            | λεπρὸς
[GREEN: exact match across all three]
...
```

## Quick Start

```bash
# 1. Install dependencies
pip install sentence-transformers chromadb anthropic requests

# 2. Download gospel data
python 01_get_data.py

# 3. Create embeddings
python 03_create_embeddings.py

# 4. Build vector database
python 04_build_vectordb.py

# 5. Find parallels
python 05_find_parallels.py

# 6. Analyze with Claude
python 06_analyze_with_llm.py

# 7. Generate synopsis
python 07_generate_synopsis.py
```

## Key Technical Decisions

### Why Verse-Level Chunking?
- Matches gospel structure (canonical verse numbers)
- Clear boundaries for comparison
- Can aggregate to larger units (pericopes) later
- Most granular analysis possible

### Why Multilingual Embeddings?
- Bible extensively represented in training data (700+ translations)
- Models learn Greek ≈ English equivalences
- No separate translation step needed
- Query in either language, find parallels in both

### Why Sentence Transformers (not LLM embeddings)?
- LLM tokens (BPE) are just input encoding, no semantic similarity
- Sentence transformers trained specifically for similarity tasks
- Fast (~1 second for entire gospel)
- Small models (~400MB vs 100GB+ for LLMs)

## Success Criteria

- **Speed**: <5 seconds per verse comparison
- **Accuracy**: >90% agreement with scholarly synopses (Aland Synopsis)
- **Coverage**: All ~2,900 verses across Matthew, Mark, Luke
- **Usability**: Students can run without programming experience

## Repository Structure

```
gospel/
├── CLAUDE.md              # This file
├── data/
│   ├── raw/              # Downloaded gospel texts
│   ├── processed/        # Structured verse data
│   └── embeddings/       # Generated embeddings
├── scripts/
│   ├── 01_get_data.py
│   ├── 02_process_data.py
│   ├── 03_create_embeddings.py
│   └── ...
├── output/
│   ├── synopses/         # Generated HTML/PDF
│   └── analysis/         # Statistics, visualizations
└── notebooks/            # Jupyter notebooks for exploration
```

## Contact

**Dave Gilbert** - Duke University
- Professor of Pharmacology and Cancer Biology
- Advisor on AI to Duke CIO

**Mark Goodacre** - Duke University
- Professor of Religious Studies
- "Jesus and the Gospels" course

---

*Version: 2.0 - Streamlined for workflow testing*
*Last updated: November 2025*
