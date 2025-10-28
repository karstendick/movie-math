# Movie Math: Semantic Movie Search

A semantic movie search application using Retrieval-Augmented Generation (RAG). Search for movies by themes, vibes, and moods - not just genres.

**🚀 [Try it live!](https://movie-math.streamlit.app/)** - Search across 21,555 movies instantly.

## Features

- **Semantic Search**: Find movies by describing themes, moods, or concepts
- **Contrastive Search**: "Like X but not Y" - find similar movies while avoiding certain aspects
- **Movie Blender**: Combine two movies to find films that blend both styles
- **Share Results**: Generate shareable URLs to share your search results with others
- **Rich Metadata**: Search includes directors, cast, genres, and plot descriptions

## Tech Stack

- **Sentence Transformers**: Local embeddings (no API costs)
- **FAISS**: Fast vector similarity search
- **Streamlit**: Interactive web interface
- **TMDb API**: Movie data and poster images

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (linting, pre-commit hooks)
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks (Optional but Recommended)

```bash
pre-commit install
```

This will automatically run Black, isort, and Flake8 on your code before each commit.

### 4. Get TMDb API Key

1. Create a free account at [TMDb](https://www.themoviedb.org/signup)
2. Go to [Settings > API](https://www.themoviedb.org/settings/api)
3. Request an API key (select "Developer" option)
4. When asked for Application URL, use: `http://localhost:8501`
5. Copy your API key (v3 auth)

### 5. Create `.env` File

```bash
# Create .env file from example
cp .env.example .env

# Edit .env and add your TMDb API key
# TMDB_API_KEY=your_api_key_here
```

### 6. Fetch and Process Movie Data

Run the setup script to fetch movies from TMDb:

```bash
python setup.py
```

This will:
- Fetch movies year-by-year from TMDb API (1874-2026)
- Fetch director and cast credits for each movie
- Save raw API responses to `data/raw/movies/{year}.json` and `data/raw/credits/{year}.json`
- Clean and process the data (minimum 100 votes required)
- Save to `data/processed/movies_clean.parquet`
- **Actual output:** 21,555 movies (with vote_count >= 100)
- **First run:** ~70 minutes (4 min for movies, 65 min for credits)
- **Subsequent runs:** Instant (uses year-based cache, resumes where left off)

The script is idempotent - run it multiple times safely:

```bash
python setup.py                    # Skip if data exists, use cache if needed
python setup.py --redownload-data  # Re-fetch data from API and regenerate everything
python setup.py --regenerate-index # Keep data, regenerate embeddings/index only
```

**How caching works:**
- Raw API data is saved year-by-year: `data/raw/movies/2024.json`, `data/raw/credits/2024.json`, etc.
- Script automatically resumes: skips years that already have cache files
- If you delete processed data but keep raw data, regeneration is instant
- Use `--force` to ignore all cache and fetch fresh data from TMDb API

### 7. Generate Embeddings and Build Index

After processing the data, generate embeddings and build the FAISS index:

```bash
python setup.py
```

If data already exists, this will:
- Load the embedding model (all-MiniLM-L6-v2, 384 dimensions)
- Generate embeddings for all 21,555 movies (~42 seconds on Apple Silicon)
- Build FAISS index for fast similarity search
- Save to `data/index/` (~76 MB total)

The script is idempotent and will skip phases that are already complete.

### 8. Try the Search Demo

Test the search functions with an interactive demo:

```bash
python search_demo.py
```

This interactive menu lets you:
- Try semantic searches with your own queries
- Test "Like X but not Y" contrastive search
- Blend two movies together
- Run example queries

Example searches to try:
- Semantic: "hopepunk", "atmospheric horror", "heist films with clever twists"
- Contrastive: Like "Lord of the Rings" but not "fantasy"
- Blending: "Blade Runner" + "Drive"

### 9. Run the Web App

Start the Streamlit web interface:

```bash
streamlit run src/app.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501)

**Features:**
- **Semantic Search tab**: Search by themes, moods, and concepts with example queries
- **Like X But Not Y tab**: Find similar movies while avoiding certain aspects
- **Movie Blender tab**: Combine two movies to discover films that blend both styles

**Optional flags:**
```bash
# Run on a different port
streamlit run src/app.py --server.port 8502

# Run in headless mode (no browser auto-open)
streamlit run src/app.py --server.headless true
```

---

## Testing

### Run Unit Tests

Unit tests run without hitting external APIs and are fast:

```bash
pytest tests/test_data_preparation_unit.py -v
```

### Run Integration Tests

Integration tests hit the TMDb API and should be run manually:

```bash
pytest tests/test_data_preparation_integration.py -v
```

### Run All Tests with Coverage

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Note:** Unit tests run automatically in pre-commit hooks and GitHub Actions. Integration tests must be run manually.

---

## Project Structure

```
movie-math/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI/CD
├── data/
│   ├── raw/                      # Raw API responses (cached, gitignored)
│   ├── processed/                # Cleaned data (gitignored)
│   └── index/                    # FAISS index and embeddings (committed)
├── src/
│   ├── data_preparation.py       # Load and clean movie data
│   ├── embeddings.py             # Generate embeddings
│   ├── search.py                 # Search functionality ✅
│   └── app.py                    # Streamlit UI ✅
├── .gitignore
├── .pre-commit-config.yaml       # Pre-commit hooks config
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── search_demo.py                # Interactive search demo ✅
├── setup.py                      # One-time data setup script
└── README.md
```

---

## Usage

### Semantic Search

Search for movies by describing what you're looking for:

- "mind-bending thrillers with plot twists"
- "heartwarming films about family"
- "atmospheric sci-fi with stunning visuals"
- "dark comedies with clever dialogue"

### Contrastive Search ("Like X But Not Y")

Find movies similar to one you like, while avoiding certain aspects:

- Like "Inception" but not "confusing"
- Like "The Godfather" but not "violent"
- Like "Eternal Sunshine" but not "sad"

### Movie Blender

Combine two movies to find films that blend both styles:

- "The Matrix" + "Her"
- "Grand Budapest Hotel" + "Knives Out"
- "Arrival" + "Eternal Sunshine"

---

## Deployment

The application is deployed and live at **[https://movie-math.streamlit.app/](https://movie-math.streamlit.app/)**

**Deployment Details:**
- **Platform**: Streamlit Community Cloud (free tier)
- **Database**: Pre-built FAISS index with 21,555 movies (committed to git)
- **No API key required**: App runs entirely off pre-built embeddings
- **Performance**: Fast search queries (~200-500ms)
- **Auto-updates**: Pushes to main branch automatically redeploy

**To deploy your own instance:**
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your forked repository
4. Set main file path: `src/app.py`
5. Deploy!
