# Movie Semantic Search - RAG Project Implementation Guide

## Project Overview

Build a semantic movie search application using Retrieval-Augmented Generation (RAG). Users can search for movies by themes, vibes, and moods - not just genres. The system uses vector embeddings to find semantically similar movies.

**Tech Stack:**
- Sentence Transformers (local embeddings)
- FAISS (local vector database)
- Streamlit (web interface)
- TMDb dataset (movies with poster images)

**Total Cost: $0** (everything runs locally)

---

## Project Structure

```
movie-rag/
├── data/
│   ├── raw/                      # Downloaded datasets
│   ├── processed/                # Cleaned data
│   └── index/                    # FAISS index and embeddings
├── src/
│   ├── data_preparation.py       # Load and clean movie data
│   ├── embeddings.py             # Generate embeddings
│   ├── search.py                 # Search functionality
│   └── app.py                    # Streamlit UI
├── requirements.txt
├── setup.py                      # One-time data setup script
└── README.md
```

---

## Step 1: Environment Setup

### Requirements (`requirements.txt`)

```
pandas>=2.0.0
numpy>=1.24.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
streamlit>=1.28.0
pillow>=10.0.0
python-dotenv>=1.0.0
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Data Acquisition

### Download TMDb Dataset

**Source:** https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

**Files needed:**
- `tmdb_5000_movies.csv` (~5MB)
- `tmdb_5000_credits.csv` (~40MB)

**Alternative:** Use Kaggle API

```bash
pip install kaggle

# Setup: Place kaggle.json in ~/.kaggle/
kaggle datasets download -d tmdb/tmdb-movie-metadata
unzip tmdb-movie-metadata.zip -d data/raw/
```

---

## Step 3: Data Preparation

### File: `src/data_preparation.py`

**Purpose:** Load, clean, and prepare movie data

**Key Functions:**

```python
def load_tmdb_data(movies_path, credits_path):
    """
    Load TMDb movies and credits data
    
    Returns:
        DataFrame with columns: id, title, overview, genres, year, 
        rating, votes, poster_url, director, cast
    """
    pass

def clean_movies(movies_df):
    """
    Filter and clean movie data:
    - Remove movies without plots (overview)
    - Remove movies without posters
    - Remove adult content
    - Parse genres from JSON strings to lists
    - Extract year from release_date
    - Filter to movies with at least 100 votes
    
    Returns:
        Cleaned DataFrame
    """
    pass

def add_poster_urls(movies_df):
    """
    Convert poster_path to full TMDb CDN URLs
    Format: https://image.tmdb.org/t/p/w500{poster_path}
    
    Returns:
        DataFrame with poster_url column
    """
    pass

def prepare_search_text(movies_df):
    """
    Create rich text for embedding by combining:
    - Title
    - Genres (space-separated)
    - Overview/plot
    
    Returns:
        DataFrame with search_text column
    """
    pass
```

**Expected Output:**
- `data/processed/movies_clean.parquet` (~15MB)
- Should have ~4,800 movies with complete data

---

## Step 4: Generate Embeddings

### File: `src/embeddings.py`

**Purpose:** Generate and save vector embeddings for all movies

**Key Functions:**

```python
def load_embedding_model():
    """
    Load Sentence Transformer model
    Model: 'all-MiniLM-L6-v2' (384 dimensions, fast)
    
    Returns:
        SentenceTransformer model
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
        numpy array of shape (n_texts, 384)
    """
    pass

def build_faiss_index(embeddings):
    """
    Build FAISS index for fast similarity search
    Use IndexFlatL2 for exact search (dataset is small enough)
    
    Args:
        embeddings: numpy array of shape (n_movies, 384)
    
    Returns:
        FAISS index
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
- `data/index/faiss_index.bin` (~7MB)
- `data/index/embeddings.npy` (~7MB)
- `data/index/movie_metadata.parquet` (~15MB)

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

def semantic_search(query, model, index, movies_df, k=10):
    """
    Basic semantic search
    
    Args:
        query: User's search query (string)
        k: Number of results to return
    
    Returns:
        DataFrame with top k movies
    """
    pass

def contrastive_search(like_movie_title, avoid_text, model, index, movies_df, k=10):
    """
    "Like X but not Y" search
    
    Implementation:
    1. Get embedding for like_movie
    2. Get embedding for avoid_text
    3. Create query vector: like_vec - 0.4 * avoid_vec
    4. Search with query vector
    5. Filter out the original movie
    
    Returns:
        DataFrame with top k movies
    """
    pass

def blend_movies(movie1_title, movie2_title, model, index, movies_df, k=10):
    """
    Find movies that combine two movies
    
    Implementation:
    1. Get embeddings for both movies
    2. Average the embeddings
    3. Search with averaged vector
    4. Filter out both original movies
    
    Returns:
        DataFrame with top k movies
    """
    pass

def filter_and_search(query, model, index, movies_df, 
                     genres=None, min_rating=None, year_range=None, k=10):
    """
    Semantic search with metadata filters
    
    Implementation:
    1. Filter movies_df by metadata first
    2. Get indices of filtered movies
    3. Create filtered FAISS index
    4. Search within filtered set
    
    Returns:
        DataFrame with top k filtered movies
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
4. 🎭 Mood Picker
5. 🔎 Advanced Search
```

**Key Components:**

```python
def setup_page_config():
    """Configure Streamlit page settings"""
    pass

def load_system():
    """Load search system with caching"""
    pass

def display_movie_grid(movies, cols=5):
    """
    Display movies in a grid with posters
    
    Each movie card shows:
    - Poster image
    - Title
    - Year and rating
    - Click for details
    """
    pass

def tab_semantic_search():
    """
    Tab 1: Basic semantic search
    
    Features:
    - Text input for query
    - Example queries as buttons
    - Result grid
    """
    pass

def tab_contrastive_search():
    """
    Tab 2: Like X But Not Y
    
    Features:
    - Dropdown to select movie (popular movies)
    - Text input for "avoid" aspects
    - Result grid
    """
    pass

def tab_movie_blender():
    """
    Tab 3: Blend two movies
    
    Features:
    - Two dropdowns for movie selection
    - Result grid
    - Explanation of blend
    """
    pass

def tab_mood_picker():
    """
    Tab 4: Pre-defined mood searches
    
    Moods:
    - Need a Good Cry
    - Brain Food
    - Pure Escapism
    - Cozy Evening
    - Date Night
    - Sunday Morning
    """
    pass

def tab_advanced_search():
    """
    Tab 5: Search with filters
    
    Features:
    - Text query
    - Genre multi-select
    - Rating slider
    - Year range slider
    - Result grid
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
```

---

## Step 7: Setup Script

### File: `setup.py`

**Purpose:** One-time script to prepare everything

```python
"""
One-time setup script
Run this once to prepare data and build index

Steps:
1. Check if data files exist
2. Load and clean data
3. Generate embeddings
4. Build FAISS index
5. Save everything

Usage:
    python setup.py
"""

def main():
    print("🎬 Movie RAG Setup")
    print("=" * 50)
    
    # Check data files
    # Load and clean
    # Generate embeddings (show progress bar)
    # Build index
    # Save
    
    print("✅ Setup complete!")
    print(f"Indexed {n_movies} movies")
    print(f"Total size: ~{total_size_mb}MB")
    print("\nRun: streamlit run src/app.py")

if __name__ == "__main__":
    main()
```

---

## Implementation Checklist

### Phase 1: Data Preparation (30 minutes)
- [ ] Download TMDb dataset
- [ ] Implement `data_preparation.py`
- [ ] Clean and filter movies
- [ ] Add poster URLs
- [ ] Save processed data
- [ ] Verify: ~4,800 movies with plots and posters

### Phase 2: Embeddings & Index (15 minutes)
- [ ] Implement `embeddings.py`
- [ ] Load Sentence Transformer model
- [ ] Generate embeddings for all movies (5-10 min runtime)
- [ ] Build FAISS index
- [ ] Save index and metadata
- [ ] Verify: Files created in `data/index/`

### Phase 3: Search Functions (30 minutes)
- [ ] Implement `search.py`
- [ ] Test semantic search
- [ ] Test contrastive search
- [ ] Test movie blending
- [ ] Test filtered search
- [ ] Verify: Queries return relevant results

### Phase 4: UI (1 hour)
- [ ] Implement basic Streamlit structure
- [ ] Create Tab 1: Semantic Search
- [ ] Create Tab 2: Like X But Not Y
- [ ] Create Tab 3: Movie Blender
- [ ] Create Tab 4: Mood Picker
- [ ] Create Tab 5: Advanced Search
- [ ] Add movie grid display
- [ ] Add custom CSS
- [ ] Test all features
- [ ] Verify: UI is responsive and looks good

### Phase 5: Polish (30 minutes)
- [ ] Add example queries
- [ ] Add loading spinners
- [ ] Add error handling
- [ ] Add README
- [ ] Test edge cases
- [ ] Deploy to Streamlit Cloud (optional)

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
- Like "Eternal Sunshine" but not "depressing"
- Like "Parasite" but not "subtitles"

**Movie Blending:**
- "The Matrix" + "Her"
- "Grand Budapest Hotel" + "Knives Out"
- "Arrival" + "Eternal Sunshine"

---

## Performance Expectations

**Setup:**
- Data preparation: ~5 minutes
- Embedding generation: ~10 minutes (CPU)
- Total setup time: ~15 minutes

**Runtime:**
- Search query: <500ms
- UI load time: <2 seconds
- Memory usage: ~500MB

**Storage:**
- Raw data: ~50MB
- Processed data: ~15MB
- Index + embeddings: ~30MB
- Total: ~100MB

---

## Deployment (Optional)

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Deploy

**Requirements:**
- Add all data files to repo (or use git-lfs for large files)
- Or re-run setup on deployment
- Free tier: 1GB RAM (sufficient for this project)

---

## Extension Ideas

**V2 Features:**
- User ratings integration
- Watchlist functionality
- Similar movies graph visualization
- Director/actor filtering
- Streaming availability (JustWatch API)
- User taste profiles (save preferences)
- Social sharing (share searches)

**Advanced RAG:**
- Hybrid search (semantic + keyword)
- Re-ranking with cross-encoder
- Query expansion
- Relevance feedback

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

**Datasets:**
- TMDb Dataset: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- TMDb Images: https://image.tmdb.org/t/p/w500/[poster_path]

**Similar Projects:**
- Search for "movie recommendation RAG" on GitHub
- Look at Streamlit gallery for UI inspiration

---

## Success Metrics

**Project is complete when:**
- [ ] Can search 4,800+ movies semantically
- [ ] Search returns relevant results in <1 second
- [ ] UI displays movie posters in grid
- [ ] All 5 tabs work correctly
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
- Use type hints
- Add docstrings to all functions
- Handle errors gracefully
- Add progress bars for long operations
- Use pathlib for file paths

**Git Commits:**
- Commit after each major component
- Clear commit messages
- Keep data files in .gitignore (too large)

Good luck! 🚀
