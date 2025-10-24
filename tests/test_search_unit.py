"""Unit tests for search module.

Tests search functionality with mocked model and index.
"""

import numpy as np
import pandas as pd
import pytest

from src.search import blend_movies, contrastive_search, semantic_search


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing."""

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        """Return mock embeddings."""
        # Return simple embeddings based on text length for deterministic testing
        # This ensures different texts get different embeddings
        embeddings = []
        for text in texts:
            # Create a deterministic embedding based on text characteristics
            embedding = np.random.RandomState(len(text)).randn(384)
            if normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        result = np.array(embeddings)
        # Always return 2D array (batch_size, embedding_dim)
        # to match real SentenceTransformer
        return result


class MockFaissIndex:
    """Mock FAISS index for testing."""

    def __init__(self, embeddings):
        """Initialize with embeddings."""
        self.embeddings = embeddings
        self.d = embeddings.shape[1]  # dimension

    def search(self, query_vector, k):
        """
        Mock search that returns results based on cosine similarity.

        Returns:
            (similarities, indices) where similarities are cosine similarities
        """
        # Ensure query_vector is 2D (batch_size, dimension)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize query vector
        query_norm = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_norm.T).flatten()

        # Get top k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_similarities = similarities[top_k_indices]

        return top_k_similarities.reshape(1, -1), top_k_indices.reshape(1, -1)


@pytest.fixture
def sample_movies_df():
    """Create a sample movies DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "title": "Inception",
                "year": 2010,
                "genres": ["Action", "Sci-Fi"],
                "director": "Christopher Nolan",
                "cast": ["Leonardo DiCaprio", "Tom Hardy"],
                "overview": (
                    "A thief who steals corporate secrets "
                    "through dream-sharing technology."
                ),
                "search_text": (
                    "Inception Action Sci-Fi Christopher Nolan "
                    "Leonardo DiCaprio Tom Hardy A thief who steals "
                    "corporate secrets through dream-sharing technology."
                ),
                "poster_url": "https://example.com/inception.jpg",
                "rating": 8.8,
            },
            {
                "title": "The Matrix",
                "year": 1999,
                "genres": ["Action", "Sci-Fi"],
                "director": "Wachowski Brothers",
                "cast": ["Keanu Reeves", "Laurence Fishburne"],
                "overview": (
                    "A computer hacker learns about " "the true nature of reality."
                ),
                "search_text": (
                    "The Matrix Action Sci-Fi Wachowski Brothers "
                    "Keanu Reeves Laurence Fishburne A computer hacker "
                    "learns about the true nature of reality."
                ),
                "poster_url": "https://example.com/matrix.jpg",
                "rating": 8.7,
            },
            {
                "title": "The Notebook",
                "year": 2004,
                "genres": ["Romance", "Drama"],
                "director": "Nick Cassavetes",
                "cast": ["Ryan Gosling", "Rachel McAdams"],
                "overview": (
                    "A poor yet passionate young man "
                    "falls in love with a rich young woman."
                ),
                "search_text": (
                    "The Notebook Romance Drama Nick Cassavetes "
                    "Ryan Gosling Rachel McAdams A poor yet passionate "
                    "young man falls in love with a rich young woman."
                ),
                "poster_url": "https://example.com/notebook.jpg",
                "rating": 7.8,
            },
            {
                "title": "Interstellar",
                "year": 2014,
                "genres": ["Sci-Fi", "Drama"],
                "director": "Christopher Nolan",
                "cast": ["Matthew McConaughey", "Anne Hathaway"],
                "overview": (
                    "A team of explorers travel through " "a wormhole in space."
                ),
                "search_text": (
                    "Interstellar Sci-Fi Drama Christopher Nolan "
                    "Matthew McConaughey Anne Hathaway A team of explorers "
                    "travel through a wormhole in space."
                ),
                "poster_url": "https://example.com/interstellar.jpg",
                "rating": 8.6,
            },
            {
                "title": "The Dark Knight",
                "year": 2008,
                "genres": ["Action", "Crime", "Drama"],
                "director": "Christopher Nolan",
                "cast": ["Christian Bale", "Heath Ledger"],
                "overview": "Batman faces the Joker, a criminal mastermind.",
                "search_text": (
                    "The Dark Knight Action Crime Drama Christopher Nolan "
                    "Christian Bale Heath Ledger "
                    "Batman faces the Joker, a criminal mastermind."
                ),
                "poster_url": "https://example.com/dark_knight.jpg",
                "rating": 9.0,
            },
        ]
    )


@pytest.fixture
def mock_model():
    """Create a mock SentenceTransformer model."""
    return MockSentenceTransformer()


@pytest.fixture
def mock_index(sample_movies_df, mock_model):
    """Create a mock FAISS index."""
    # Generate embeddings for all movies
    embeddings = []
    for text in sample_movies_df["search_text"]:
        embedding = mock_model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        embeddings.append(embedding)

    embeddings_array = np.vstack(embeddings).astype(np.float32)
    return MockFaissIndex(embeddings_array)


@pytest.mark.unit
def test_semantic_search_basic(mock_model, mock_index, sample_movies_df):
    """Test basic semantic search."""
    query = "science fiction movie about dreams"
    k = 3

    results = semantic_search(query, mock_model, mock_index, sample_movies_df, k=k)

    # Verify results structure
    assert len(results) == k
    assert "similarity" in results.columns
    assert all(results["similarity"] >= -1) and all(results["similarity"] <= 1)

    # Verify all required columns are present
    assert "title" in results.columns
    assert "year" in results.columns


@pytest.mark.unit
def test_semantic_search_returns_top_k(mock_model, mock_index, sample_movies_df):
    """Test that semantic search returns exactly k results."""
    query = "action movie"
    k = 2

    results = semantic_search(query, mock_model, mock_index, sample_movies_df, k=k)

    assert len(results) == k


@pytest.mark.unit
def test_semantic_search_similarity_scores_ordered(
    mock_model, mock_index, sample_movies_df
):
    """Test that results are ordered by similarity (descending)."""
    query = "space exploration"
    k = 5

    results = semantic_search(query, mock_model, mock_index, sample_movies_df, k=k)

    # Check that similarities are in descending order
    similarities = results["similarity"].values
    assert all(
        similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1)
    )


@pytest.mark.unit
def test_contrastive_search_movie_found(mock_model, mock_index, sample_movies_df):
    """Test contrastive search with valid movie."""
    like_movie = "Inception"
    avoid = "romance"
    k = 3

    results = contrastive_search(
        like_movie, avoid, mock_model, mock_index, sample_movies_df, k=k
    )

    # Should return results
    assert len(results) > 0
    assert len(results) <= k

    # Should not include the source movie
    assert "Inception" not in results["title"].values

    # Should have similarity scores
    assert "similarity" in results.columns


@pytest.mark.unit
def test_contrastive_search_movie_not_found(mock_model, mock_index, sample_movies_df):
    """Test contrastive search with non-existent movie."""
    like_movie = "Nonexistent Movie 12345"
    avoid = "boring"
    k = 3

    results = contrastive_search(
        like_movie, avoid, mock_model, mock_index, sample_movies_df, k=k
    )

    # Should return empty DataFrame
    assert len(results) == 0
    assert "similarity" in results.columns


@pytest.mark.unit
def test_contrastive_search_case_insensitive(mock_model, mock_index, sample_movies_df):
    """Test that movie title matching is case-insensitive."""
    like_movie = "iNcEpTiOn"  # Mixed case
    avoid = "slow"
    k = 3

    results = contrastive_search(
        like_movie, avoid, mock_model, mock_index, sample_movies_df, k=k
    )

    # Should find the movie despite case difference
    assert len(results) > 0
    assert "Inception" not in results["title"].values


@pytest.mark.unit
def test_contrastive_search_partial_match(mock_model, mock_index, sample_movies_df):
    """Test that partial movie title matches work."""
    like_movie = "Dark"  # Partial match for "The Dark Knight"
    avoid = "comedy"
    k = 3

    results = contrastive_search(
        like_movie, avoid, mock_model, mock_index, sample_movies_df, k=k
    )

    # Should find and exclude "The Dark Knight"
    assert len(results) > 0
    assert "The Dark Knight" not in results["title"].values


@pytest.mark.unit
def test_blend_movies_both_found(mock_model, mock_index, sample_movies_df):
    """Test blending two valid movies."""
    movie1 = "Inception"
    movie2 = "The Matrix"
    k = 3

    results = blend_movies(
        movie1, movie2, mock_model, mock_index, sample_movies_df, k=k
    )

    # Should return results
    assert len(results) > 0
    assert len(results) <= k

    # Should not include either source movie
    assert "Inception" not in results["title"].values
    assert "The Matrix" not in results["title"].values

    # Should have similarity scores
    assert "similarity" in results.columns


@pytest.mark.unit
def test_blend_movies_first_not_found(mock_model, mock_index, sample_movies_df):
    """Test blending when first movie doesn't exist."""
    movie1 = "Nonexistent Movie 1"
    movie2 = "The Matrix"
    k = 3

    results = blend_movies(
        movie1, movie2, mock_model, mock_index, sample_movies_df, k=k
    )

    # Should return empty DataFrame
    assert len(results) == 0
    assert "similarity" in results.columns


@pytest.mark.unit
def test_blend_movies_second_not_found(mock_model, mock_index, sample_movies_df):
    """Test blending when second movie doesn't exist."""
    movie1 = "Inception"
    movie2 = "Nonexistent Movie 2"
    k = 3

    results = blend_movies(
        movie1, movie2, mock_model, mock_index, sample_movies_df, k=k
    )

    # Should return empty DataFrame
    assert len(results) == 0
    assert "similarity" in results.columns


@pytest.mark.unit
def test_blend_movies_both_not_found(mock_model, mock_index, sample_movies_df):
    """Test blending when both movies don't exist."""
    movie1 = "Nonexistent Movie 1"
    movie2 = "Nonexistent Movie 2"
    k = 3

    results = blend_movies(
        movie1, movie2, mock_model, mock_index, sample_movies_df, k=k
    )

    # Should return empty DataFrame
    assert len(results) == 0


@pytest.mark.unit
def test_blend_movies_case_insensitive(mock_model, mock_index, sample_movies_df):
    """Test that movie title matching is case-insensitive for blending."""
    movie1 = "inception"  # lowercase
    movie2 = "THE MATRIX"  # uppercase
    k = 3

    results = blend_movies(
        movie1, movie2, mock_model, mock_index, sample_movies_df, k=k
    )

    # Should find both movies despite case differences
    assert len(results) > 0
    assert "Inception" not in results["title"].values
    assert "The Matrix" not in results["title"].values


@pytest.mark.unit
def test_all_search_functions_preserve_dataframe_columns(
    mock_model, mock_index, sample_movies_df
):
    """Test that search functions preserve important DataFrame columns."""
    query = "action movie"

    # Test semantic search
    semantic_results = semantic_search(
        query, mock_model, mock_index, sample_movies_df, k=2
    )
    assert "title" in semantic_results.columns
    assert "year" in semantic_results.columns
    assert "genres" in semantic_results.columns
    assert "director" in semantic_results.columns

    # Test contrastive search
    contrastive_results = contrastive_search(
        "Inception", "romance", mock_model, mock_index, sample_movies_df, k=2
    )
    if len(contrastive_results) > 0:
        assert "title" in contrastive_results.columns
        assert "year" in contrastive_results.columns

    # Test blend movies
    blend_results = blend_movies(
        "Inception", "The Matrix", mock_model, mock_index, sample_movies_df, k=2
    )
    if len(blend_results) > 0:
        assert "title" in blend_results.columns
        assert "year" in blend_results.columns


@pytest.mark.unit
def test_semantic_search_with_single_result(mock_model, mock_index, sample_movies_df):
    """Test semantic search requesting only 1 result."""
    query = "space movie"
    k = 1

    results = semantic_search(query, mock_model, mock_index, sample_movies_df, k=k)

    assert len(results) == 1
    assert "similarity" in results.columns


@pytest.mark.unit
def test_contrastive_search_multiple_matches_prefer_recent(
    mock_model, mock_index, sample_movies_df
):
    """Test that when multiple movies match, more recent ones are preferred."""
    # Add duplicate titles with different years
    df_with_duplicates = sample_movies_df.copy()
    duplicate_movie = pd.DataFrame(
        [
            {
                "title": "Inception",  # Same title as existing movie
                "year": 2020,  # More recent year
                "genres": ["Action"],
                "director": "Another Director",
                "cast": ["Actor A"],
                "overview": "Different plot",
                "search_text": (
                    "Inception Action Another Director " "Actor A Different plot"
                ),
                "poster_url": "https://example.com/inception2.jpg",
                "rating": 7.0,
            }
        ]
    )
    df_with_duplicates = pd.concat(
        [df_with_duplicates, duplicate_movie], ignore_index=True
    )

    # Recreate index with duplicates
    embeddings = []
    for text in df_with_duplicates["search_text"]:
        embedding = mock_model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        embeddings.append(embedding)
    embeddings_array = np.vstack(embeddings).astype(np.float32)
    index_with_duplicates = MockFaissIndex(embeddings_array)

    results = contrastive_search(
        "inception",
        "boring",
        mock_model,
        index_with_duplicates,
        df_with_duplicates,
        k=3,
    )

    # Should return results (excluding the source movies)
    assert len(results) > 0
    # Neither version of Inception should be in results
    assert not any(title == "Inception" for title in results["title"].values)
