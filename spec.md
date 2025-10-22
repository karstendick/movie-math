# Movie Math: Semantic Search using RAG

## Project Overview

Build a semantic movie search application using Retrieval-Augmented Generation (RAG) in order to learn the concept and build up my portfolio. Users can search for movies by themes, vibes, and moods - not just genres. The system uses vector embeddings to find semantically similar movies.

**Tech Stack:**
- Sentence Transformers (local embeddings)
- FAISS (local vector database)
- Streamlit (web interface)
- TMDb API (movies with poster images)

**Total Cost: $0** (everything runs locally)

---

## Project Structure

```
movie-math/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI/CD
├── data/
│   ├── raw/
│   │   ├── movies/               # Year-based movie cache (1874.json - 2025.json)
│   │   └── credits/              # Year-based credits cache (1874.json - 2025.json)
│   ├── processed/                # Cleaned data (movies_clean.parquet)
│   └── index/                    # FAISS index and embeddings
├── src/
│   ├── data_preparation.py       # Load and clean movie data
│   ├── embeddings.py             # Generate embeddings
│   ├── search.py                 # Search functionality
│   └── app.py                    # Streamlit UI
├── .gitignore
├── .pre-commit-config.yaml       # Pre-commit hooks config
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── setup.py                      # One-time data setup script
└── README.md
```

---

## Step 1: Environment Setup

### Create `.gitignore`

```
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Environment
.env

# Data (raw and processed - regenerate with setup.py)
data/raw/
data/processed/

# OS
.DS_Store
.vscode/
.idea/

# Note: data/index/ SHOULD be committed for deployment
# Index files (~60-80MB) are under GitHub's 100MB limit
```

### Requirements (`requirements.txt`)

```
pandas~=2.0.0
numpy~=1.24.0
sentence-transformers~=2.2.0
faiss-cpu~=1.7.4
streamlit~=1.28.0
pillow~=10.0.0
python-dotenv~=1.0.0
requests~=2.31.0

# Note: Using ~= (compatible release) instead of >= for stability
# ~=2.0.0 allows 2.0.x and 2.x.x but not 3.0.0
# Prevents breaking changes from major version updates
```

### Dev Requirements (`requirements-dev.txt`)

```
# Linting and code quality
black~=23.0.0
flake8~=6.0.0
isort~=5.12.0
mypy~=1.5.0

# Pre-commit hooks
pre-commit~=3.3.0

# Future: Testing (when added in V2)
# pytest~=7.4.0
# pytest-cov~=4.1.0
```

### Pre-commit Configuration (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=102400']  # 100MB limit
```

### GitHub Actions Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt

    - name: Run Black
      run: black --check src/ setup.py

    - name: Run isort
      run: isort --check-only --profile black src/ setup.py

    - name: Run Flake8
      run: flake8 src/ setup.py --max-line-length=88 --extend-ignore=E203,W503

    # Future: Add test job when tests are implemented
    # test:
    #   runs-on: ubuntu-latest
    #   steps:
    #   - uses: actions/checkout@v3
    #   - name: Set up Python
    #     uses: actions/setup-python@v4
    #     with:
    #       python-version: '3.11'
    #   - name: Install dependencies
    #     run: |
    #       pip install -r requirements.txt
    #       pip install -r requirements-dev.txt
    #   - name: Run tests
    #     run: pytest tests/
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

---

## Step 2: Data Acquisition

### Get TMDb API Key

**Source:** https://www.themoviedb.org/settings/api

**Steps:**
1. Create free TMDb account at https://www.themoviedb.org/signup
2. Go to Settings > API
3. Request API key (select "Developer" option)
4. Copy your API key (v3 auth)

**Setup:**
```bash
# Create .env file in project root
echo "TMDB_API_KEY=your_api_key_here" > .env
```

**API Details:**
- Free tier: 50 requests/second
- Unlimited daily requests
- Use `/discover/movie` endpoint to fetch movies
- Fetch year-by-year (1874-2026) to bypass 500-page limit per query
- Minimum votes: 100 (configurable via MIN_VOTE_COUNT constant)
- Sort by: popularity.desc (single sort method per year)
- **Actual output: 21,555 movies** with complete data (1874-2025)
- **Runtime: ~70 minutes** (4 min movies, 65 min credits)

---

## Step 3: Data Preparation

### File: `src/data_preparation.py`

**Purpose:** Load, clean, and prepare movie data

**Logging:**
```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log key events:
# - Number of movies fetched from API
# - Data cleaning steps (how many removed, why)
# - Warnings for missing data
```

**Configuration Constants:**
```python
# Configurable thresholds - adjust these to tune dataset quality
MIN_VOTE_COUNT = 100  # Minimum votes to include a movie
MIN_VOTE_AVERAGE = 0  # Minimum rating (0 = no filter)
# Lower MIN_VOTE_COUNT includes more indie/art films
# Higher MIN_VOTE_COUNT focuses on popular/well-known movies
# 100 votes = good balance: keeps 90% of movies, filters truly obscure films
```

**Key Functions:**

```python
def fetch_movies_from_api(api_key, min_votes=MIN_VOTE_COUNT, use_cache=True):
    """
    Fetch movies from TMDb API using year-based pagination.

    Uses /discover/movie endpoint with year-based filtering.
    Strategy: Fetch movies year-by-year sorted by popularity
    - Fetches from 1874 (first motion pictures) to current_year + 1
    - Uses primary_release_year filter to bypass 500-page limit per query
    - Sorts by popularity.desc to get most relevant movies first
    - Each year can have up to 500 pages (unlikely for most years)

    This approach ensures comprehensive coverage across all film history
    without hitting TMDb's pagination limits.

    Caching:
    - Save raw API responses to data/raw/movies/{year}.json
    - Cache allows re-running data preparation without re-fetching from API
    - Skip years that already have cache files (unless use_cache=False)
    - Enables resumability: can stop/restart without losing progress

    Error handling:
    - Retry failed requests up to 3 times with exponential backoff
    - Handle rate limiting (429 status) by waiting and retrying
    - Validate API key before starting (test with simple request)
    - Continue with other years if one year fails
    - Log progress every 10 years

    Args:
        api_key: TMDb API key from .env
        min_votes: Minimum vote count filter (default MIN_VOTE_COUNT=100)
        use_cache: If True, load from cache if available (default True)

    Returns:
        DataFrame with columns: id, title, overview, genres, release_date,
        rating, votes, poster_path, popularity

    Raises:
        ValueError: If API key is invalid
    """
    pass

def fetch_movie_credits(api_key, movie_id):
    """
    Fetch director and cast for a specific movie

    Uses /movie/{movie_id}/credits endpoint

    Extraction:
    - Director: First crew member with job == "Director"
    - Cast: First 5 actors by billing order (sorted by 'order' field)
      - order: 0 = top billing, order: 1 = second billing, etc.
      - TMDb API returns cast pre-sorted by billing order
      - Simply take cast[:5]

    Error handling:
    - Retry failed requests up to 3 times
    - Skip movie if credits unavailable (log warning)
    - Continue processing other movies on failure

    Returns:
        dict with 'director' (string) and 'cast' (list of 5 names)
        Example: {'director': 'Christopher Nolan', 'cast': ['Leonardo DiCaprio', 'Joseph Gordon-Levitt', ...]}
        Returns None if credits unavailable
    """
    pass

def load_tmdb_data_from_api(api_key, use_cache=True):
    """
    Complete data loading from TMDb API

    1. Fetch all movies using fetch_movies_from_api()
    2. Fetch credits for each movie (with progress bar)
    3. Combine into single DataFrame

    Caching:
    - Saves credits to data/raw/credits/{year}.json for re-use
    - Groups movies by year for organized caching
    - Skip years with cached credits (unless use_cache=False)
    - Enables resumability: can stop/restart credits fetching

    Args:
        api_key: TMDb API key
        use_cache: If True, load from cache if available (default True)

    Returns:
        DataFrame with columns: id, title, overview, genres, release_date,
        rating, votes, poster_path, director, cast, year
    """
    pass

def clean_movies(movies_df):
    """
    Filter and clean movie data:
    - Remove movies without plots (overview)
    - Remove movies without posters
    - Remove adult content
    - Parse genres from JSON to list of names:
      Input: [{"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}]
      Output: ["Drama", "Thriller"]
      Use: [g['name'] for g in movie['genres']]
    - Extract year from release_date (YYYY-MM-DD → YYYY)
    - Filter using MIN_VOTE_COUNT constant (defined above)

    Returns:
        Cleaned DataFrame with genres as list of strings
    """
    pass

def add_poster_urls(movies_df):
    """
    Convert poster_path to full TMDb CDN URLs
    Format: https://image.tmdb.org/t/p/w500{poster_path}

    Add fallback for missing posters:
    - If poster_path is null/empty, use placeholder
    - Placeholder options: 🎬 emoji, data URI, or static image

    Returns:
        DataFrame with poster_url column
    """
    pass

def prepare_search_text(movies_df):
    """
    Create rich text for embedding by combining:
    - Title
    - Genres (join list with spaces: ["Drama", "Thriller"] → "Drama Thriller")
    - Director name
    - Top 5 cast members (join with spaces)
    - Overview/plot

    Example output:
    "Inception Science Fiction Action Thriller Christopher Nolan Leonardo DiCaprio Joseph Gordon-Levitt Ellen Page Tom Hardy Marion Cotillard A thief who steals corporate secrets..."

    This enables searches like:
    - "Tarantino films" (director match)
    - "movies with Tom Hanks" (cast match)
    - "sci-fi with great acting" (genre + overview)

    Returns:
        DataFrame with search_text column
    """
    pass
```

**Expected Output:**
- `data/processed/movies_clean.parquet` (~25-40MB)
- Should have ~10,000-15,000 movies with complete data
- Includes classics (The Godfather, Casablanca, etc.) and recent films (2020s)

---

## Step 4: Generate Embeddings

### File: `src/embeddings.py`

**Purpose:** Generate and save vector embeddings for all movies

**Logging:**
```python
import logging
logger = logging.getLogger(__name__)

# Log key events:
# - Model loading progress
# - Embedding generation progress (with progress bar)
# - Index build completion
# - File save operations
```

**Configuration:**
```python
# Configurable model - easy to experiment with different options
MODEL_NAME = "all-MiniLM-L6-v2"  # Default: fast and good quality

# Alternative options:
# MODEL_NAME = "all-mpnet-base-v2"  # Higher quality, 2x slower, 2x storage
#   - Best for: Portfolio/demo quality
#   - Tradeoff: 20 min setup vs 10 min, 60MB embeddings vs 30MB
# MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"  # Optimized for queries
#   - Best for: Query-document matching
#   - Tradeoff: Movie blending might be slightly worse
```

**Key Functions:**

```python
def load_embedding_model(model_name=MODEL_NAME):
    """
    Load Sentence Transformer model

    Args:
        model_name: Model to load (default: from MODEL_NAME constant)

    Returns:
        SentenceTransformer model

    Model will be cached in ~/.cache/torch/sentence_transformers/
    First download may take a minute
    """
    pass

def generate_embeddings(texts, model, batch_size=32):
    """
    Generate embeddings for list of texts

    Args:
        texts: List of strings to embed
        model: SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        numpy array of shape (n_texts, embedding_dim)
        Embeddings are L2-normalized for cosine similarity
    """
    pass

def build_faiss_index(embeddings):
    """
    Build FAISS index for fast similarity search

    Use IndexFlatIP (Inner Product) for cosine similarity:
    - Requires L2-normalized embeddings (done in generate_embeddings)
    - Inner product of normalized vectors = cosine similarity
    - Better for sentence embeddings than Euclidean distance (L2)

    Dataset is small enough (~10-15K movies) for exact search.

    Args:
        embeddings: numpy array of shape (n_movies, embedding_dim)
        - Must be L2-normalized
        - 384 dims for MiniLM models
        - 768 dims for MPNet models

    Returns:
        FAISS IndexFlatIP
    """
    pass

def save_index_and_embeddings(index, embeddings, movies_df, output_dir):
    """
    Save FAISS index, embeddings, and movie metadata to disk

    Files to create:
    - faiss_index.bin
    - embeddings.npy
    - movie_metadata.parquet
    """
    pass
```

**Expected Output:**
- `data/index/faiss_index.bin` (~15-20MB)
- `data/index/embeddings.npy` (~15-20MB)
- `data/index/movie_metadata.parquet` (~25-40MB)

---

## Step 5: Search Implementation

### File: `src/search.py`

**Purpose:** Implement various search strategies

**Key Functions:**

```python
def load_search_system(index_dir):
    """
    Load FAISS index, embeddings, and metadata

    Returns:
        (model, index, movies_df)
    """
    pass

def semantic_search(query, model, index, movies_df, k=100):
    """
    Basic semantic search

    Args:
        query: User's search query (string)
        k: Number of results to return (default 100 for pagination)

    Returns:
        DataFrame with top k movies, including 'similarity' column from FAISS
        With IndexFlatIP, scores are already cosine similarities (range: -1 to 1)
        Higher score = better match
    """
    pass

def contrastive_search(like_movie_title, avoid_text, model, index, movies_df, k=100):
    """
    "Like X but not Y" search

    Implementation:
    1. Get embedding for like_movie
    2. Get embedding for avoid_text
    3. Create query vector: like_vec - 0.5 * avoid_vec
       - Weight of 0.5 balances between keeping movie's vibe and avoiding unwanted aspects
       - Too low (0.1): Barely affects results
       - Too high (1.0): Might over-correct and lose the original movie's character
       - 0.5 is a reasonable middle ground (can be tuned later if needed)
    4. Search with query vector
    5. Filter out the original movie

    Returns:
        DataFrame with top k movies (default 100 for pagination)
    """
    pass

def blend_movies(movie1_title, movie2_title, model, index, movies_df, k=100):
    """
    Find movies that combine two movies

    Implementation:
    1. Get embeddings for both movies
    2. Average the embeddings
    3. Search with averaged vector
    4. Filter out both original movies

    Returns:
        DataFrame with top k movies (default 100 for pagination)
    """
    pass
```

---

## Step 6: Streamlit UI

### File: `src/app.py`

**Purpose:** Web interface for movie search

**UI Structure:**

```
Title: 🎬 Semantic Movie Search
Subtitle: Find movies by vibe, mood, and themes

Tabs:
1. 🔍 Semantic Search
2. 🎯 Like X But Not Y
3. 🎬 Movie Blender
```

**Key Components:**

```python
def setup_page_config():
    """Configure Streamlit page settings"""
    pass

@st.cache_resource
def load_system():
    """
    Load search system with caching

    Error handling:
    - Check if required index files exist:
      - data/index/faiss_index.bin
      - data/index/embeddings.npy
      - data/index/movie_metadata.parquet
    - If any missing, show error message and stop:
      st.error("Index files not found. Please run: python setup.py")
      st.stop()
    - Load model, index, and metadata

    Returns:
        (model, index, movies_df)
    """
    pass

def display_movie_grid(movies, cols=5, results_per_page=20, show_similarity=True):
    """
    Display movies in a grid with posters and pagination

    Each movie card shows:
    - Poster image
    - Title and year
    - Similarity score (if show_similarity=True)
      - Convert cosine similarity to percentage: (similarity + 1) / 2 * 100
      - Cosine range: -1 to 1, convert to 0-100%
      - Display as: "🎯 94% match"
    - Rating (⭐ 8.8)
    - Genres
    - Director name
    - Click for details/expanded view

    Pagination:
    - Fetch more results from search (k=100)
    - Display results_per_page at a time (default 20)
    - Use st.pagination or page number selector
    - Shows "Page 1 of 5" with prev/next buttons
    """
    pass

def tab_semantic_search():
    """
    Tab 1: Basic semantic search

    Features:
    - Text input for query
    - Search button disabled until query has text
    - Example queries as buttons (pre-fill search box)
    - Result grid
    """
    pass

def tab_contrastive_search():
    """
    Tab 2: Like X But Not Y

    Features:
    - Text input with autocomplete for movie selection
      - Shows: movie title, year, and tiny poster thumbnail
      - Searches as user types
      - Helps disambiguate (e.g., "Inception (2010)" vs other movies)
    - Text input for "avoid" aspects (required field)
    - Search button disabled until both fields filled
    - Result grid
    """
    pass

def tab_movie_blender():
    """
    Tab 3: Blend two movies

    Features:
    - Two text inputs with autocomplete for movie selection
      - Shows: movie title, year, and tiny poster thumbnail
    - Search button disabled until both movies selected
    - Result grid
    - Explanation of blend
    """
    pass
```

**CSS Styling:**

```python
def inject_custom_css():
    """
    Add custom CSS for:
    - Movie card hover effects
    - Poster image borders and shadows
    - Grid layout
    - Responsive design
    """
    pass

def display_footer():
    """
    Display footer with TMDb attribution

    Required text:
    "This product uses the TMDb API but is not endorsed or certified by TMDb.
    Movie data and posters provided by TMDb."

    Include link to: https://www.themoviedb.org
    """
    pass
```

---

## Step 7: Setup Script

### File: `setup.py`

**Purpose:** Idempotent setup script to prepare everything

**Design Principle: Idempotency**
- Can be run multiple times safely
- Checks if files exist before regenerating
- Allows re-downloading data with `--redownload-data` flag
- Allows regenerating embeddings/index only with `--regenerate-index` flag
- Smart incremental updates where possible

```python
"""
Idempotent setup script
Can be run multiple times safely - only regenerates what's needed

Steps:
1. Check if processed data exists (skip if exists, unless --redownload-data)
2. Check if index exists (skip if exists, unless --regenerate-index)
3. Load and clean data from TMDb API
4. Generate embeddings
5. Build FAISS index
6. Save everything

Usage:
    python setup.py                    # Run setup, skip existing files
    python setup.py --redownload-data  # Re-fetch data and regenerate everything
    python setup.py --regenerate-index # Keep data, regenerate embeddings/index only

Flags:
    --redownload-data: Re-fetch movie data from TMDb API and regenerate everything
    --regenerate-index: Regenerate embeddings and index only (keep existing data)
"""

import argparse
from pathlib import Path

def check_existing_files():
    """
    Check what files already exist

    Returns:
        dict with keys: has_processed_data, has_index, has_embeddings
    """
    pass

def main():
    parser = argparse.ArgumentParser(description='Movie RAG Setup')
    parser.add_argument('--redownload-data', action='store_true',
                       help='Re-fetch data from TMDb API and regenerate everything')
    parser.add_argument('--regenerate-index', action='store_true',
                       help='Regenerate embeddings and index only (keep existing data)')
    args = parser.parse_args()

    print("🎬 Movie RAG Setup")
    print("=" * 50)

    existing = check_existing_files()

    # Determine what needs to be done
    if args.redownload_data:
        print("🔄 Re-downloading data from TMDb API and regenerating everything...")
        fetch_data = True
        generate_index = True
    elif args.regenerate_index:
        print("🔄 Regenerate index mode: Keeping data, regenerating index...")
        fetch_data = False
        generate_index = True
    else:
        fetch_data = not existing['has_processed_data']
        generate_index = not existing['has_index']

        if not fetch_data and not generate_index:
            print("✅ All files exist. Nothing to do!")
            print("   Use --redownload-data to re-fetch from TMDb API")
            print("   Use --regenerate-index to regenerate embeddings/index only")
            return

    # Execute setup steps
    if fetch_data:
        print("\n📥 Fetching movie data from TMDb API...")
        try:
            # Validate API key first
            # Load and clean data
            # Show progress bar
            print(f"✅ Fetched {n_movies} movies")
        except ValueError as e:
            print(f"❌ Error: {e}")
            print("   Check your TMDb API key in .env file")
            return
        except requests.RequestException as e:
            print(f"❌ Network error: {e}")
            print("   Check your internet connection and try again")
            return
    else:
        print("\n⏭️  Skipping data fetch (already exists)")
        # Load existing data

    if generate_index:
        print("\n🧮 Generating embeddings...")
        # Generate embeddings (show progress bar)
        print("\n🔨 Building FAISS index...")
        # Build index
        # Save everything
        print("✅ Index built successfully")
    else:
        print("\n⏭️  Skipping index generation (already exists)")

    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print(f"📊 Indexed {n_movies} movies")
    print(f"💾 Total size: ~{total_size_mb}MB")
    print(f"\n🚀 Run: streamlit run src/app.py")

if __name__ == "__main__":
    main()
```

---

## Implementation Checklist

### Phase 0: Project Setup (10 minutes)
- [x] Create project structure (directories: data/raw, data/processed, data/index, src)
- [x] Create .gitignore file
- [x] Create requirements.txt
- [x] Create requirements-dev.txt (linting tools)
- [x] Create pre-commit config
- [x] Create GitHub Actions workflow
- [x] Create virtual environment and install dependencies
- [x] Initialize git repository
- [x] Install pre-commit hooks

### Phase 1: Data Preparation ✅ COMPLETE
- [x] Get TMDb API key and add to .env
- [x] Implement `data_preparation.py` with year-based fetching
- [x] Implement API fetching functions (bypasses 500-page limit)
- [x] Implement idempotent `setup.py` with `--redownload-data` and `--regenerate-index` flags
- [x] Implement year-based raw data caching (data/raw/movies/{year}.json)
- [x] Implement year-based credits caching (data/raw/credits/{year}.json)
- [x] Run `python setup.py` to fetch movies from TMDb API (~70 min runtime)
- [x] Verify: 21,555 movies with plots and posters (1874-2025)
- [x] Test idempotency: Run `python setup.py` again (should skip existing files)
- [x] Unit and integration tests passing
- [x] Increased MIN_VOTE_COUNT to 100 for better quality

### Phase 2: Embeddings & Index ✅ COMPLETE
- [x] Implement `embeddings.py`
- [x] Load Sentence Transformer model
- [x] Generate embeddings for all movies (~42 seconds on Apple Silicon)
- [x] Build FAISS index
- [x] Save index and metadata
- [x] Verify: Files created in `data/index/` (75.7 MB total)
- [x] Optimize with lazy imports (28x faster CLI startup)

### Phase 3: Search Functions (30 minutes)
- [ ] Implement `search.py`
- [ ] Test semantic search
- [ ] Test contrastive search
- [ ] Test movie blending
- [ ] Verify: Queries return relevant results

### Phase 4: UI (1 hour)
- [ ] Implement basic Streamlit structure
- [ ] Create Tab 1: Semantic Search
- [ ] Create Tab 2: Like X But Not Y
- [ ] Create Tab 3: Movie Blender
- [ ] Add movie grid display with similarity scores
- [ ] Add custom CSS
- [ ] Test all features
- [ ] Verify: UI is responsive and looks good

### Phase 5: Polish (30 minutes)
- [ ] Add example queries
- [ ] Add loading spinners
- [ ] Add error handling
- [ ] Add TMDb attribution footer
- [ ] Add README
- [ ] Test edge cases

### Phase 6: Deployment (10 minutes)
- [ ] Commit data/index/ files to git
- [ ] Push to GitHub
- [ ] Deploy to Streamlit Cloud
- [ ] (Optional) Add TMDb API key as Streamlit secret for data refreshes
- [ ] Test deployed app
- [ ] Verify all features work in production

---

## Example Queries to Test

**Semantic Search:**
- "mind-bending thrillers with plot twists"
- "heartwarming films about family"
- "atmospheric sci-fi with stunning visuals"
- "dark comedies with clever dialogue"
- "movies about the cost of ambition"

**Contrastive Search:**
- Like "Inception" but not "confusing"
- Like "The Godfather" but not "violent"
- Like "Eternal Sunshine" but not "sad"
- Like "The Matrix" but not "action-heavy"

**Movie Blending:**
- "The Matrix" + "Her"
- "Grand Budapest Hotel" + "Knives Out"
- "Arrival" + "Eternal Sunshine"

---

## Performance Expectations

**Setup:**
- TMDb API data fetch: ~15-20 minutes
- Data preparation: ~5 minutes
- Embedding generation: ~20-25 minutes (CPU)
- Total setup time: ~40-50 minutes

**Runtime:**
- Search query: <500ms
- UI load time: <2 seconds
- Memory usage: ~500MB

**Storage:**
- Raw data: ~10-15MB (cached API responses in data/raw/)
- Processed data: ~25-40MB
- Index + embeddings: ~60-80MB
- Total: ~95-135MB

---

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Deploy

**Requirements:**
- Commit data/index/ files to git repository
  - Size: ~60-80MB (under GitHub's 100MB file limit)
  - Necessary for app to work on deployment
- Add TMDb API key as Streamlit secret (optional, only needed for data refreshes)
- Free tier: 1GB RAM (sufficient for this project)

**Add API key as secret (optional):**
1. Go to Streamlit Cloud app settings
2. Add secret: `TMDB_API_KEY = "your_key_here"`
3. Only needed if you want to refresh data using `setup.py --refresh`

---

## Future Enhancements

See [v2-features.md](v2-features.md) for a complete list of potential features and improvements to implement after the MVP is complete.

**Highlights:**
- Accessibility improvements (high priority)
- Testing & quality (unit tests, integration tests, code coverage)
- Mood Picker tab with pre-defined searches
- Advanced RAG techniques (hybrid search, re-ranking, query expansion)

---

## Troubleshooting

**Common Issues:**

1. **Out of memory during embedding generation**
   - Reduce batch_size in generate_embeddings()
   - Process in chunks and save incrementally

2. **Slow search queries**
   - Verify FAISS index is loaded correctly
   - Check if using GPU (should be CPU for this scale)
   - Ensure embeddings are float32, not float64

3. **Poor search results**
   - Check if search_text includes genres
   - Verify embeddings were generated correctly
   - Try different similarity metrics (cosine vs L2)

4. **Missing posters**
   - Some TMDb poster_paths may be invalid
   - Add fallback placeholder image
   - Check poster_url construction

5. **Streamlit deployment fails**
   - Check requirements.txt versions
   - Verify all data files are accessible
   - Check memory usage (upgrade plan if needed)

---

## Resources

**Documentation:**
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- Streamlit: https://docs.streamlit.io/
- TMDb API: https://developers.themoviedb.org/

**Images:**
- TMDb Images: https://image.tmdb.org/t/p/w500/[poster_path]

**Similar Projects:**
- Search for "movie recommendation RAG" on GitHub
- Look at Streamlit gallery for UI inspiration

---

## Success Metrics

**Project is complete when:**
- [ ] Can search 10,000+ movies semantically (classics to modern)
- [ ] Database includes well-known films from all eras (The Godfather, Inception, etc.)
- [ ] Search returns relevant results in <1 second
- [ ] UI displays movie posters in grid with similarity scores
- [ ] All 3 tabs work correctly (Semantic Search, Contrastive Search, Movie Blender)
- [ ] Example queries return expected results
- [ ] Code is documented and clean
- [ ] Can demo to someone in 5 minutes

---

## Notes for Claude Code

**Implementation Priority:**
1. Start with data_preparation.py - get clean data first
2. Then embeddings.py - generate index
3. Then search.py - test search functions in isolation
4. Finally app.py - build UI

**Testing Strategy:**
- Test each module independently before integration
- Use small sample (100 movies) for initial testing
- Verify search relevance manually with known movies

**Code Style:**
- Use Black for formatting (88 char line length)
- Use isort for import sorting (Black-compatible profile)
- Use Flake8 for linting (ignoring E203, W503 for Black compatibility)
- Use type hints
- Add docstrings to all functions
- Handle errors gracefully
- Add progress bars for long operations
- Use pathlib for file paths

**Development Workflow:**
- Pre-commit hooks run automatically on commit
- Run `black src/ setup.py` to format code
- Run `isort src/ setup.py --profile black` to sort imports
- Run `flake8 src/ setup.py` to check linting
- GitHub Actions runs on all PRs and pushes to main

**Git Commits:**
- Commit after each major component
- Clear commit messages
- Keep processed data in .gitignore (regenerate with setup.py)
- Commit index files to git for deployment (~60-80MB, under GitHub limit)
- Pre-commit hooks ensure code quality before commit
