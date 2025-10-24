"""Integration tests for data preparation (requires TMDb API key).

Run these manually with: pytest tests/test_data_preparation_integration.py

These tests hit the actual TMDb API and should not run in CI.
"""

import os

import pytest
from dotenv import load_dotenv

from src.data_preparation import (
    fetch_movie_credits,
    fetch_movies_from_api,
    validate_api_key,
)

load_dotenv()


@pytest.fixture
def api_key():
    """Get TMDb API key from environment."""
    key = os.getenv("TMDB_API_KEY")
    if not key:
        pytest.skip("TMDB_API_KEY not set in environment")
    return key


@pytest.mark.integration
def test_api_key_validation(api_key):
    """Test API key validation with real API."""
    assert validate_api_key(api_key) is True


@pytest.mark.integration
def test_invalid_api_key_rejected():
    """Test that invalid API key raises ValueError."""
    with pytest.raises(ValueError, match="Invalid TMDb API key"):
        validate_api_key("invalid_key_12345")


@pytest.mark.integration
def test_empty_api_key_rejected():
    """Test that empty API key raises ValueError."""
    with pytest.raises(ValueError, match="API key is missing"):
        validate_api_key("")


@pytest.mark.integration
def test_fetch_small_sample(api_key):
    """Test fetching movies from API."""
    # Note: This will fetch ALL movies meeting criteria
    # For a quick test, we just verify the function works
    df = fetch_movies_from_api(api_key, min_votes=50)

    assert len(df) > 0, "Should fetch some movies"
    assert "title" in df.columns
    assert "genres" in df.columns
    assert "overview" in df.columns
    assert all(df["votes"] >= 50), "All movies should have >= 50 votes"


@pytest.mark.integration
def test_fetch_movie_credits_inception(api_key):
    """Test fetching credits for Inception."""
    credits = fetch_movie_credits(api_key, 27205)  # Inception movie ID

    assert credits is not None
    assert credits["director"] == "Christopher Nolan"
    assert "Leonardo DiCaprio" in credits["cast"]
    assert len(credits["cast"]) <= 5, "Should have at most 5 cast members"


@pytest.mark.integration
def test_fetch_movie_credits_invalid_id(api_key):
    """Test fetching credits for invalid movie ID."""
    # Using a very high ID that shouldn't exist
    credits = fetch_movie_credits(api_key, 99999999)

    # Should return None or empty dict for invalid ID
    assert credits is None or credits["director"] is None
