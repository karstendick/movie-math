"""Unit tests for data preparation module (no API calls)."""

import pandas as pd

from src.data_preparation import add_poster_urls, clean_movies, prepare_search_text


def test_clean_movies():
    """Test data cleaning filters invalid movies."""
    sample_data = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "Valid Movie",
                "overview": "A test movie with description",
                "genres": ["Action", "Drama"],
                "release_date": "2020-01-01",
                "rating": 7.5,
                "votes": 100,
                "poster_path": "/test.jpg",
                "director": "Test Director",
                "cast": ["Actor 1", "Actor 2"],
            },
            {
                "id": 2,
                "title": "No Overview",
                "overview": "",  # Should be filtered
                "genres": ["Comedy"],
                "release_date": "2021-01-01",
                "rating": 6.0,
                "votes": 100,
                "poster_path": "/test2.jpg",
                "director": "Director 2",
                "cast": [],
            },
            {
                "id": 3,
                "title": "Low Votes",
                "overview": "Has overview",
                "genres": ["Drama"],
                "release_date": "2021-01-01",
                "rating": 8.0,
                "votes": 30,  # Below MIN_VOTE_COUNT (50)
                "poster_path": "/test3.jpg",
                "director": "Director 3",
                "cast": ["Actor 3"],
            },
        ]
    )

    cleaned = clean_movies(sample_data)

    assert len(cleaned) == 1, "Should keep only 1 valid movie"
    assert cleaned.iloc[0]["title"] == "Valid Movie"
    assert "year" in cleaned.columns, "Should add year column"


def test_add_poster_urls():
    """Test poster URL generation."""
    sample_data = pd.DataFrame(
        [
            {"poster_path": "/abc123.jpg"},
            {"poster_path": None},
            {"poster_path": ""},
        ]
    )

    result = add_poster_urls(sample_data)

    assert "poster_url" in result.columns
    assert result.iloc[0]["poster_url"] == "https://image.tmdb.org/t/p/w500/abc123.jpg"
    assert "placeholder" in result.iloc[1]["poster_url"].lower()
    assert "placeholder" in result.iloc[2]["poster_url"].lower()


def test_prepare_search_text():
    """Test search text generation."""
    sample_data = pd.DataFrame(
        [
            {
                "title": "Test Movie",
                "genres": ["Action", "Drama"],
                "director": "Christopher Nolan",
                "cast": ["Actor 1", "Actor 2", "Actor 3"],
                "overview": "An epic story about time.",
            }
        ]
    )

    result = prepare_search_text(sample_data)

    assert "search_text" in result.columns
    search_text = result.iloc[0]["search_text"]

    # Verify all components are included
    assert "Test Movie" in search_text
    assert "Action" in search_text
    assert "Drama" in search_text
    assert "Christopher Nolan" in search_text
    assert "Actor 1" in search_text
    assert "An epic story" in search_text


def test_prepare_search_text_with_missing_fields():
    """Test search text handles missing fields gracefully."""
    sample_data = pd.DataFrame(
        [
            {
                "title": "Minimal Movie",
                "genres": [],
                "director": None,
                "cast": [],
                "overview": "Just an overview",
            }
        ]
    )

    result = prepare_search_text(sample_data)

    search_text = result.iloc[0]["search_text"]
    assert "Minimal Movie" in search_text
    assert "Just an overview" in search_text
    assert isinstance(search_text, str)
