# Movie Math: Semantic Movie Search

A semantic movie search application using Retrieval-Augmented Generation (RAG). Search for movies by themes, vibes, and moods - not just genres.

## Features

- **Semantic Search**: Find movies by describing themes, moods, or concepts
- **Contrastive Search**: "Like X but not Y" - find similar movies while avoiding certain aspects
- **Movie Blender**: Combine two movies to find films that blend both styles
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

### 6. Run Setup Script

```bash
# This will fetch movie data from TMDb and build the search index
# First run takes ~40-50 minutes (fetching data + generating embeddings)
python setup.py
```

The setup script is idempotent - you can run it multiple times safely:

```bash
python setup.py              # Skip existing files
python setup.py --force      # Regenerate everything
python setup.py --refresh    # Re-fetch data from API
```

### 7. Launch the App

```bash
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

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
