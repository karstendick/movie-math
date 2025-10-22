"""
Search module for Movie Math.

Implements various search strategies:
- Semantic search: Find movies by themes, vibes, and moods
- Contrastive search: "Like X but not Y" search
- Movie blending: Find movies that combine two movies

Note: Heavy dependencies (sentence_transformers, faiss) are imported lazily
within functions to avoid slow startup times when not needed.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Type hints only, not runtime imports
    import faiss
    from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_search_system(
    index_dir: Path,
) -> Tuple["SentenceTransformer", "faiss.IndexFlatIP", pd.DataFrame]:
    """
    Load FAISS index, embeddings, and metadata.

    Args:
        index_dir: Directory containing index files

    Returns:
        (model, index, movies_df)

    Raises:
        FileNotFoundError: If required files are missing
    """
    # Import the load function from embeddings module
    from src.embeddings import load_search_system as load_system_impl

    return load_system_impl(index_dir)


def semantic_search(
    query: str,
    model: "SentenceTransformer",
    index: "faiss.IndexFlatIP",
    movies_df: pd.DataFrame,
    k: int = 100,
) -> pd.DataFrame:
    """
    Basic semantic search.

    Args:
        query: User's search query (string)
        model: SentenceTransformer model
        index: FAISS index
        movies_df: DataFrame with movie metadata
        k: Number of results to return (default 100 for pagination)

    Returns:
        DataFrame with top k movies, including 'similarity' column from FAISS
        With IndexFlatIP, scores are already cosine similarities (range: -1 to 1)
        Higher score = better match
    """
    logger.info(f"Semantic search for: '{query}'")

    # Generate embedding for query
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )

    # Search FAISS index
    # Returns (distances, indices) where distances are cosine similarities
    similarities, indices = index.search(query_embedding.astype(np.float32), k)

    # Extract results
    top_indices = indices[0]  # First row (we only have one query)
    top_similarities = similarities[0]  # First row

    # Get movies from dataframe
    results_df = movies_df.iloc[top_indices].copy()

    # Add similarity scores
    results_df["similarity"] = top_similarities

    logger.info(f"Found {len(results_df)} results")
    logger.info(
        f"Top match: '{results_df.iloc[0]['title']}' (score: {top_similarities[0]:.4f})"
    )

    return results_df


def contrastive_search(
    like_movie_title: str,
    avoid_text: str,
    model: "SentenceTransformer",
    index: "faiss.IndexFlatIP",
    movies_df: pd.DataFrame,
    k: int = 100,
) -> pd.DataFrame:
    """
    "Like X but not Y" search.

    Implementation:
    1. Get embedding for like_movie
    2. Get embedding for avoid_text
    3. Create query vector: like_vec - 1.0 * avoid_vec
       - Weight of 1.0 provides strong contrast to avoid unwanted aspects
       - Lower weights (0.3-0.5): Results too similar to source movie
       - Weight of 1.0: Balanced push away from unwanted aspects
       - Higher weights (>1.5): May over-correct and lose movie's character
    4. Search with query vector
    5. Filter out the original movie

    Args:
        like_movie_title: Title of the movie to base search on
        avoid_text: Text describing what to avoid
        model: SentenceTransformer model
        index: FAISS index
        movies_df: DataFrame with movie metadata
        k: Number of results to return (default 100 for pagination)

    Returns:
        DataFrame with top k movies (default 100 for pagination)
    """
    logger.info(f"Contrastive search: Like '{like_movie_title}' but not '{avoid_text}'")

    # Find the movie by title (case-insensitive)
    # Try exact match first
    movie_matches = movies_df[
        movies_df["title"].str.lower() == like_movie_title.lower()
    ]

    # If no exact match, try partial match
    if len(movie_matches) == 0:
        movie_matches = movies_df[
            movies_df["title"]
            .str.lower()
            .str.contains(like_movie_title.lower(), na=False)
        ]

    if len(movie_matches) == 0:
        logger.warning(f"Movie not found: '{like_movie_title}'")
        # Return empty dataframe with same columns
        result_df = movies_df.iloc[:0].copy()
        result_df["similarity"] = pd.Series(dtype=float)
        return result_df

    # Use the first match (most popular based on dataset order)
    # If multiple matches, prefer more recent movies by sorting by year descending
    if len(movie_matches) > 1 and "year" in movie_matches.columns:
        movie_matches = movie_matches.sort_values("year", ascending=False)

    source_movie = movie_matches.iloc[0]
    source_movie_id = source_movie.name  # Index of the movie in dataframe
    logger.info(
        f"Found source movie: '{source_movie['title']}' "
        f"({source_movie.get('year', 'N/A')})"
    )

    # Get embedding for the source movie from its search_text
    source_text = source_movie["search_text"]
    like_embedding = model.encode(
        [source_text], convert_to_numpy=True, normalize_embeddings=True
    )[0]

    # Get embedding for avoid text
    avoid_embedding = model.encode(
        [avoid_text], convert_to_numpy=True, normalize_embeddings=True
    )[0]

    # Create contrastive query vector: like - 1.0 * avoid
    query_vector = like_embedding - 1.0 * avoid_embedding

    # Normalize the query vector for cosine similarity
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Search FAISS index
    # Request k+1 to account for filtering out the source movie
    similarities, indices = index.search(
        query_vector.reshape(1, -1).astype(np.float32), k + 1
    )

    # Extract results
    top_indices = indices[0]
    top_similarities = similarities[0]

    # Get movies from dataframe
    results_df = movies_df.iloc[top_indices].copy()

    # Add similarity scores
    results_df["similarity"] = top_similarities

    # Filter out the original source movie
    results_df = results_df[results_df.index != source_movie_id]

    # Limit to k results
    results_df = results_df.head(k)

    logger.info(f"Found {len(results_df)} results (excluding source movie)")
    if len(results_df) > 0:
        logger.info(
            f"Top match: '{results_df.iloc[0]['title']}' "
            f"(score: {results_df.iloc[0]['similarity']:.4f})"
        )

    return results_df


def blend_movies(
    movie1_title: str,
    movie2_title: str,
    model: "SentenceTransformer",
    index: "faiss.IndexFlatIP",
    movies_df: pd.DataFrame,
    k: int = 100,
) -> pd.DataFrame:
    """
    Find movies that combine two movies.

    Implementation:
    1. Get embeddings for both movies
    2. Average the embeddings
    3. Search with averaged vector
    4. Filter out both original movies

    Args:
        movie1_title: Title of first movie
        movie2_title: Title of second movie
        model: SentenceTransformer model
        index: FAISS index
        movies_df: DataFrame with movie metadata
        k: Number of results to return (default 100 for pagination)

    Returns:
        DataFrame with top k movies (default 100 for pagination)
    """
    logger.info(f"Blending movies: '{movie1_title}' + '{movie2_title}'")

    # Find both movies by title (case-insensitive)
    # Try exact match first
    movie1_matches = movies_df[movies_df["title"].str.lower() == movie1_title.lower()]
    if len(movie1_matches) == 0:
        movie1_matches = movies_df[
            movies_df["title"].str.lower().str.contains(movie1_title.lower(), na=False)
        ]

    movie2_matches = movies_df[movies_df["title"].str.lower() == movie2_title.lower()]
    if len(movie2_matches) == 0:
        movie2_matches = movies_df[
            movies_df["title"].str.lower().str.contains(movie2_title.lower(), na=False)
        ]

    if len(movie1_matches) == 0:
        logger.warning(f"Movie 1 not found: '{movie1_title}'")
        result_df = movies_df.iloc[:0].copy()
        result_df["similarity"] = pd.Series(dtype=float)
        return result_df

    if len(movie2_matches) == 0:
        logger.warning(f"Movie 2 not found: '{movie2_title}'")
        result_df = movies_df.iloc[:0].copy()
        result_df["similarity"] = pd.Series(dtype=float)
        return result_df

    # Use the first match for each, prefer more recent if multiple matches
    if len(movie1_matches) > 1 and "year" in movie1_matches.columns:
        movie1_matches = movie1_matches.sort_values("year", ascending=False)
    if len(movie2_matches) > 1 and "year" in movie2_matches.columns:
        movie2_matches = movie2_matches.sort_values("year", ascending=False)

    movie1 = movie1_matches.iloc[0]
    movie2 = movie2_matches.iloc[0]
    movie1_id = movie1.name
    movie2_id = movie2.name

    logger.info(f"Found movie 1: '{movie1['title']}' ({movie1.get('year', 'N/A')})")
    logger.info(f"Found movie 2: '{movie2['title']}' ({movie2.get('year', 'N/A')})")

    # Get embeddings for both movies from their search_text
    movie1_text = movie1["search_text"]
    movie2_text = movie2["search_text"]

    embedding1 = model.encode(
        [movie1_text], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    embedding2 = model.encode(
        [movie2_text], convert_to_numpy=True, normalize_embeddings=True
    )[0]

    # Average the embeddings
    query_vector = (embedding1 + embedding2) / 2.0

    # Normalize the averaged vector for cosine similarity
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Search FAISS index
    # Request k+2 to account for filtering out both source movies
    similarities, indices = index.search(
        query_vector.reshape(1, -1).astype(np.float32), k + 2
    )

    # Extract results
    top_indices = indices[0]
    top_similarities = similarities[0]

    # Get movies from dataframe
    results_df = movies_df.iloc[top_indices].copy()

    # Add similarity scores
    results_df["similarity"] = top_similarities

    # Filter out both original movies
    results_df = results_df[~results_df.index.isin([movie1_id, movie2_id])]

    # Limit to k results
    results_df = results_df.head(k)

    logger.info(f"Found {len(results_df)} results (excluding source movies)")
    if len(results_df) > 0:
        logger.info(
            f"Top match: '{results_df.iloc[0]['title']}' "
            f"(score: {results_df.iloc[0]['similarity']:.4f})"
        )

    return results_df
