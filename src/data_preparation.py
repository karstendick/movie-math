"""
Data preparation module for Movie Math.

Handles fetching, cleaning, and preparing movie data from TMDb API.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurable thresholds - adjust these to tune dataset quality
MIN_VOTE_COUNT = 100  # Minimum votes to include a movie
MIN_VOTE_AVERAGE = 0  # Minimum rating (0 = no filter)
# Lower MIN_VOTE_COUNT includes more indie/art films
# Higher MIN_VOTE_COUNT focuses on popular/well-known movies


def validate_api_key(api_key: str) -> bool:
    """
    Validate TMDb API key with a test request.

    Args:
        api_key: TMDb API key to validate

    Returns:
        True if API key is valid; False, otherwise

    Raises:
        ValueError: If API key is invalid or missing
    """
    if not api_key:
        raise ValueError(
            "TMDb API key is missing. Please set TMDB_API_KEY in .env file"
        )

    test_url = "https://api.themoviedb.org/3/configuration"
    params = {"api_key": api_key}

    try:
        response = requests.get(test_url, params=params, timeout=10)
        if response.status_code == 401:
            raise ValueError("Invalid TMDb API key. Please check your .env file")
        response.raise_for_status()
        logger.info("TMDb API key validated successfully")
        return True
    except requests.RequestException as e:
        raise ValueError(f"Failed to validate API key: {e}")


def _save_raw_data(
    data: List[Dict[str, Any]], subdirectory: str, filename: str
) -> None:
    """
    Save raw API data to data/raw/{subdirectory}/ directory.

    Args:
        data: List of dictionaries containing raw API responses
        subdirectory: Subdirectory under data/raw/ (e.g., 'movies', 'credits')
        filename: Name of the file to save (e.g., '2024.json')
    """
    raw_dir = Path("data/raw") / subdirectory
    raw_dir.mkdir(parents=True, exist_ok=True)
    filepath = raw_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved raw data to {filepath}")


def _load_raw_data(subdirectory: str, filename: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load raw API data from data/raw/{subdirectory}/ directory.

    Args:
        subdirectory: Subdirectory under data/raw/ (e.g., 'movies', 'credits')
        filename: Name of the file to load (e.g., '2024.json')

    Returns:
        List of dictionaries if file exists, None otherwise
    """
    filepath = Path("data/raw") / subdirectory / filename

    if not filepath.exists():
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded raw data from {filepath}")
    return data


def _fetch_movies_for_year(
    api_key: str, year: int, min_votes: int = MIN_VOTE_COUNT
) -> List[Dict[str, Any]]:
    """
    Fetch all movies for a specific year from TMDb API.

    Uses /discover/movie endpoint with year filtering and popularity sorting.
    Stops when: no results OR 500 page limit reached.

    Args:
        api_key: TMDb API key
        year: Release year to fetch (e.g., 2024)
        min_votes: Minimum vote count filter

    Returns:
        List of movie dictionaries

    Raises:
        requests.RequestException: If network request fails after retries
    """
    base_url = "https://api.themoviedb.org/3/discover/movie"
    movies = []
    page = 1
    max_pages = 500  # TMDb API limit

    while page <= max_pages:
        params = {
            "api_key": api_key,
            "sort_by": "popularity.desc",
            "vote_count.gte": min_votes,
            "primary_release_year": year,
            "include_adult": "false",
            "include_video": "false",
            "language": "en-US",
            "page": page,
        }

        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, params=params, timeout=10)

                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Rate limited. Waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                # Check if we've reached the end
                results = data.get("results", [])
                if not results:
                    logger.debug(f"  No more results at page {page}")
                    return movies

                # Extract movies from response
                for movie in results:
                    movies.append(
                        {
                            "id": movie.get("id"),
                            "title": movie.get("title"),
                            "overview": movie.get("overview"),
                            "genres": movie.get("genre_ids", []),
                            "release_date": movie.get("release_date"),
                            "rating": movie.get("vote_average"),
                            "votes": movie.get("vote_count"),
                            "poster_path": movie.get("poster_path"),
                            "popularity": movie.get("popularity"),
                        }
                    )

                # Success - break retry loop
                break

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to fetch {year} page {page} after "
                        f"{max_retries} attempts: {e}"
                    )
                    raise
                wait_time = 2**attempt
                logger.warning(f"Request failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # Small delay to be respectful to API
        time.sleep(0.1)
        page += 1

    if page > max_pages:
        logger.warning(f"  {year}: Reached TMDb API limit of {max_pages} pages")

    return movies


def fetch_movies_from_api(
    api_key: str, min_votes: int = MIN_VOTE_COUNT, use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch ALL movies from TMDb API that meet quality criteria.

    Uses /discover/movie endpoint with year-based pagination.
    Strategy: Fetch movies year-by-year sorted by popularity
    - Fetches from 1874 (first films) to current year + 1
    - Uses primary_release_year filter to bypass 500-page limit
    - Sorts by popularity.desc to get most relevant movies first

    Caching:
    - Save raw API responses to data/raw/movies/{year}.json
    - Cache allows re-running data preparation without re-fetching from API
    - Skip years that already have cache files (unless use_cache=False)

    Error handling:
    - Retry failed requests up to 3 times with exponential backoff
    - Handle rate limiting (429 status) by waiting and retrying
    - Validate API key before starting (test with simple request)
    - Clear error messages for network failures
    - Continue with other years if one year fails

    Args:
        api_key: TMDb API key from .env
        min_votes: Minimum vote count filter (default MIN_VOTE_COUNT)
        use_cache: If True, load from cache if available (default True)

    Returns:
        DataFrame with columns: id, title, overview, genres, release_date,
        rating, votes, poster_path, popularity

    Raises:
        ValueError: If API key is invalid
    """
    import datetime

    # Validate API key first
    validate_api_key(api_key)

    current_year = datetime.datetime.now().year
    start_year = 1874  # First motion pictures
    end_year = current_year + 1  # Include upcoming releases

    all_movies = []
    seen_ids = set()

    logger.info(
        f"Fetching movies from {start_year} to {end_year} "
        f"(vote_count >= {min_votes})..."
    )

    years_fetched = 0
    years_cached = 0
    years_total = end_year - start_year + 1

    for year in range(start_year, end_year + 1):
        cache_filename = f"{year}.json"

        # Check cache first
        if use_cache:
            cached_year_data = _load_raw_data("movies", cache_filename)
            if cached_year_data is not None:
                years_cached += 1
                # Add to all_movies, deduplicating by ID
                for movie in cached_year_data:
                    movie_id = movie.get("id")
                    if movie_id and movie_id not in seen_ids:
                        seen_ids.add(movie_id)
                        all_movies.append(movie)
                continue

        # Fetch from API
        try:
            year_movies = _fetch_movies_for_year(api_key, year, min_votes)

            if year_movies:
                years_fetched += 1
                logger.info(f"  {year}: Fetched {len(year_movies)} movies")

                # Save to cache
                _save_raw_data(year_movies, "movies", cache_filename)

                # Add to all_movies, deduplicating by ID
                for movie in year_movies:
                    movie_id = movie.get("id")
                    if movie_id and movie_id not in seen_ids:
                        seen_ids.add(movie_id)
                        all_movies.append(movie)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {year}, skipping: {e}")
            continue

        # Progress update every 10 years
        if year % 10 == 0:
            progress = year - start_year + 1
            logger.info(
                f"Progress: {progress}/{years_total} years processed, "
                f"{len(all_movies)} unique movies so far"
            )

    logger.info(
        f"Fetched {len(all_movies)} unique movies from TMDb API "
        f"({years_fetched} years fetched, {years_cached} years from cache)"
    )

    # Convert to DataFrame
    df = pd.DataFrame(all_movies)

    # Map genre IDs to names
    df = _map_genre_ids_to_names(df, api_key)

    return df


def _map_genre_ids_to_names(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Map genre IDs to genre names using TMDb genre list.

    Args:
        df: DataFrame with genre_ids column
        api_key: TMDb API key

    Returns:
        DataFrame with genres column containing list of genre names
    """
    # Fetch genre list from TMDb
    genre_url = "https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": api_key, "language": "en-US"}

    try:
        response = requests.get(genre_url, params=params, timeout=10)
        response.raise_for_status()
        genre_data = response.json()

        # Create genre ID to name mapping
        genre_map = {g["id"]: g["name"] for g in genre_data.get("genres", [])}

        # Map genre IDs to names
        df["genres"] = df["genres"].apply(
            lambda ids: [genre_map.get(gid, "Unknown") for gid in ids] if ids else []
        )

        logger.info("Mapped genre IDs to names")

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch genre list: {e}. Using genre IDs instead.")
        df["genres"] = df["genres"].apply(lambda ids: ids if ids else [])

    return df


def fetch_movie_credits(api_key: str, movie_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch director and cast for a specific movie.

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
        Example: {'director': 'Christopher Nolan',
                  'cast': ['Leonardo DiCaprio', ...]}
        Returns None if credits unavailable
    """
    credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
    params = {"api_key": api_key}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(credits_url, params=params, timeout=10)

            # Handle rate limiting
            if response.status_code == 429:
                wait_time = 2**attempt
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            # Extract director
            director = None
            for crew_member in data.get("crew", []):
                if crew_member.get("job") == "Director":
                    director = crew_member.get("name")
                    break

            # Extract top 5 cast members (already sorted by billing order)
            cast = [actor.get("name") for actor in data.get("cast", [])[:5]]

            return {"director": director, "cast": cast}

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                logger.warning(f"Failed to fetch credits for movie {movie_id}: {e}")
                return None
            time.sleep(2**attempt)

    return None


def load_tmdb_data_from_api(api_key: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Complete data loading from TMDb API.

    1. Fetch all movies using fetch_movies_from_api()
    2. Fetch credits for each movie (with progress bar)
    3. Combine into single DataFrame

    Caching:
    - Saves credits to data/raw/credits_raw.json for re-use
    - Skip credits API calls if cache exists (unless use_cache=False)

    Args:
        api_key: TMDb API key
        use_cache: If True, load from cache if available (default True)

    Returns:
        DataFrame with columns: id, title, overview, genres, year,
        rating, votes, poster_path, director, cast
    """
    # Fetch movies
    logger.info("Starting movie data fetch from TMDb API...")
    movies_df = fetch_movies_from_api(api_key, use_cache=use_cache)

    # Extract year from release_date for grouping
    movies_df["year"] = pd.to_datetime(
        movies_df["release_date"], errors="coerce"
    ).dt.year

    # Group movies by year
    movies_by_year = movies_df.groupby("year", dropna=False)

    # Build credits mapping from cache and API
    all_credits_map = {}

    logger.info(f"Fetching credits for {len(movies_df)} movies...")
    total_years = len(movies_by_year)
    years_processed = 0
    years_from_cache = 0
    years_from_api = 0

    for year, year_movies in movies_by_year:
        # Skip invalid years
        if pd.isna(year):
            year_label = "unknown"
        else:
            year_label = str(int(year))

        cache_filename = f"{year_label}.json"

        # Try to load from cache
        if use_cache:
            cached_credits = _load_raw_data("credits", cache_filename)
            if cached_credits is not None:
                years_from_cache += 1
                # Add to credits map
                for credit in cached_credits:
                    all_credits_map[credit["movie_id"]] = credit
                years_processed += 1
                continue

        # Fetch from API
        year_credits = []
        for _, movie in year_movies.iterrows():
            movie_id = movie["id"]
            credits = fetch_movie_credits(api_key, movie_id)

            if credits:
                credit_entry = {
                    "movie_id": movie_id,
                    "director": credits["director"],
                    "cast": credits["cast"],
                }
            else:
                credit_entry = {"movie_id": movie_id, "director": None, "cast": []}

            year_credits.append(credit_entry)
            all_credits_map[movie_id] = credit_entry

            # Small delay to be respectful to API
            time.sleep(0.1)

        # Save year credits to cache
        if year_credits:
            _save_raw_data(year_credits, "credits", cache_filename)
            years_from_api += 1

        years_processed += 1
        logger.info(
            f"  Progress: {years_processed}/{total_years} years, "
            f"{len(all_credits_map)} movies with credits"
        )

    logger.info(
        f"Credits loaded: {years_from_cache} years from cache, "
        f"{years_from_api} years from API"
    )

    # Apply credits to dataframe
    directors = []
    casts = []
    for _, row in movies_df.iterrows():
        movie_id = row["id"]
        if movie_id in all_credits_map:
            directors.append(all_credits_map[movie_id]["director"])
            casts.append(all_credits_map[movie_id]["cast"])
        else:
            directors.append(None)
            casts.append([])

    movies_df["director"] = directors
    movies_df["cast"] = casts

    logger.info(f" Loaded {len(movies_df)} movies with credits")

    return movies_df


def clean_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and clean movie data:
    - Remove movies without plots (overview)
    - Remove movies without posters
    - Remove adult content
    - Parse genres from JSON to list of names (already done in fetch)
    - Extract year from release_date (YYYY-MM-DD � YYYY)
    - Filter using MIN_VOTE_COUNT constant (defined above)

    Returns:
        Cleaned DataFrame with genres as list of strings
    """
    initial_count = len(movies_df)
    logger.info(f"Cleaning {initial_count} movies...")

    # Remove movies without overview
    movies_df = movies_df[movies_df["overview"].notna() & (movies_df["overview"] != "")]
    logger.info(f"  Removed {initial_count - len(movies_df)} movies without overview")
    initial_count = len(movies_df)

    # Remove movies without posters
    movies_df = movies_df[
        movies_df["poster_path"].notna() & (movies_df["poster_path"] != "")
    ]
    logger.info(f"  Removed {initial_count - len(movies_df)} movies without poster")
    initial_count = len(movies_df)

    # Extract year from release_date
    movies_df["year"] = pd.to_datetime(
        movies_df["release_date"], errors="coerce"
    ).dt.year
    movies_df = movies_df[movies_df["year"].notna()]
    logger.info(
        f"  Removed {initial_count - len(movies_df)} movies with invalid release date"
    )

    # Filter by minimum vote count
    initial_count = len(movies_df)
    movies_df = movies_df[movies_df["votes"] >= MIN_VOTE_COUNT]
    logger.info(
        f"  Removed {initial_count - len(movies_df)} movies with "
        f"<{MIN_VOTE_COUNT} votes"
    )

    # Remove movies without director (optional - keep movies without director)
    # Commenting out to be more inclusive
    # initial_count = len(movies_df)
    # movies_df = movies_df[movies_df["director"].notna()]
    # logger.info(f"  Removed {initial_count - len(movies_df)} movies without director")

    # Reset index
    movies_df = movies_df.reset_index(drop=True)

    logger.info(f" Cleaned dataset: {len(movies_df)} movies remaining")

    return movies_df


def add_poster_urls(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert poster_path to full TMDb CDN URLs.
    Format: https://image.tmdb.org/t/p/w500{poster_path}

    Add fallback for missing posters:
    - If poster_path is null/empty, use placeholder
    - Placeholder: Use a simple placeholder URL

    Returns:
        DataFrame with poster_url column
    """
    base_url = "https://image.tmdb.org/t/p/w500"
    placeholder = "https://via.placeholder.com/500x750?text=No+Poster"

    movies_df["poster_url"] = movies_df["poster_path"].apply(
        lambda path: f"{base_url}{path}" if pd.notna(path) and path else placeholder
    )

    logger.info("Added poster URLs")

    return movies_df


def prepare_search_text(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rich text for embedding by combining:
    - Title
    - Genres (join list with spaces: ["Drama", "Thriller"] � "Drama Thriller")
    - Director name
    - Top 5 cast members (join with spaces)
    - Overview/plot

    Example output:
    "Inception Science Fiction Action Thriller Christopher Nolan
    Leonardo DiCaprio Joseph Gordon-Levitt Ellen Page Tom Hardy
    Marion Cotillard A thief who steals corporate secrets..."

    This enables searches like:
    - "Tarantino films" (director match)
    - "movies with Tom Hanks" (cast match)
    - "sci-fi with great acting" (genre + overview)

    Returns:
        DataFrame with search_text column
    """

    def create_search_text(row):
        parts = []

        # Add title
        if pd.notna(row["title"]):
            parts.append(row["title"])

        # Add genres
        if row["genres"] and len(row["genres"]) > 0:
            parts.append(" ".join(row["genres"]))

        # Add director
        if pd.notna(row["director"]):
            parts.append(row["director"])

        # Add cast
        if row["cast"] and len(row["cast"]) > 0:
            parts.append(" ".join(row["cast"]))

        # Add overview
        if pd.notna(row["overview"]):
            parts.append(row["overview"])

        return " ".join(parts)

    movies_df["search_text"] = movies_df.apply(create_search_text, axis=1)

    logger.info("Created search text for embeddings")

    return movies_df


def save_processed_data(movies_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save processed movie data to parquet file.

    Args:
        movies_df: Cleaned and processed movie DataFrame
        output_path: Path to save parquet file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies_df.to_parquet(output_path, index=False)
    logger.info(f" Saved processed data to {output_path}")


def load_processed_data(input_path: Path) -> pd.DataFrame:
    """
    Load processed movie data from parquet file.

    Args:
        input_path: Path to parquet file

    Returns:
        DataFrame with processed movie data
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Processed data not found at {input_path}")

    movies_df = pd.read_parquet(input_path)
    logger.info(f" Loaded {len(movies_df)} movies from {input_path}")

    return movies_df
