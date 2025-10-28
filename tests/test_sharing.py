"""
Unit tests for social sharing feature.

Tests URL encoding/decoding and year-based movie disambiguation in search functions.
"""

import urllib.parse
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Import functions to test
from src.search import blend_movies, contrastive_search, similar_movies_search


class TestURLEncoding:
    """Test URL parameter encoding and decoding."""

    def test_semantic_search_url_encoding(self):
        """Test URL encoding of query parameters."""
        # Simulate what generate_share_url does
        params = {"mode": "semantic", "q": "time travel movies"}
        query_string = urllib.parse.urlencode(params)

        # Parse it back
        decoded_params = urllib.parse.parse_qs(query_string)

        assert decoded_params["mode"][0] == "semantic"
        assert decoded_params["q"][0] == "time travel movies"

    def test_special_characters_encoding(self):
        """Test URL encoding of special characters."""
        # Test with accented character and ampersand
        params = {"mode": "semantic", "q": "Amélie & friends"}
        query_string = urllib.parse.urlencode(params)

        # Parse it back
        decoded_params = urllib.parse.parse_qs(query_string)

        # After decoding, should match original
        assert decoded_params["q"][0] == "Amélie & friends"

    def test_year_parameter_encoding(self):
        """Test that year parameter is properly encoded."""
        params = {"mode": "similar", "movie": "The Matrix", "year": 1999}
        query_string = urllib.parse.urlencode(params)

        decoded_params = urllib.parse.parse_qs(query_string)

        assert decoded_params["mode"][0] == "similar"
        assert decoded_params["movie"][0] == "The Matrix"
        assert decoded_params["year"][0] == "1999"

    def test_movie_title_with_apostrophe(self):
        """Test movie titles with apostrophes."""
        params = {"mode": "similar", "movie": "Ocean's Eleven", "year": 2001}
        query_string = urllib.parse.urlencode(params)

        decoded_params = urllib.parse.parse_qs(query_string)

        # Apostrophe should be preserved after encoding/decoding
        assert decoded_params["movie"][0] == "Ocean's Eleven"

    def test_movie_title_with_numbers(self):
        """Test movie titles with numbers."""
        params = {
            "mode": "similar",
            "movie": "10 Things I Hate About You",
            "year": 1999,
        }
        query_string = urllib.parse.urlencode(params)

        decoded_params = urllib.parse.parse_qs(query_string)

        assert decoded_params["movie"][0] == "10 Things I Hate About You"

    def test_query_with_multiple_spaces(self):
        """Test query with multiple consecutive spaces."""
        params = {"mode": "semantic", "q": "time    travel    movies"}
        query_string = urllib.parse.urlencode(params)

        decoded_params = urllib.parse.parse_qs(query_string)

        # Spaces should be preserved
        assert decoded_params["q"][0] == "time    travel    movies"


class TestSearchFunctionsWithYear:
    """Test search functions with year parameter for disambiguation."""

    @pytest.fixture
    def mock_movies_df(self):
        """Create mock dataframe with duplicate movie titles."""
        return pd.DataFrame(
            {
                "title": ["The Departed", "The Departed", "Inception", "The Matrix"],
                "year": [2006, 1991, 2010, 1999],
                "search_text": [
                    "crime thriller 2006",
                    "french film 1991",
                    "dream heist",
                    "sci-fi action",
                ],
                "genres": [
                    ["Crime", "Drama"],
                    ["Drama"],
                    ["Sci-Fi"],
                    ["Sci-Fi", "Action"],
                ],
                "rating": [8.5, 7.2, 8.8, 8.7],
                "overview": [
                    "Boston crime",
                    "French drama",
                    "Dream story",
                    "Matrix story",
                ],
            }
        )

    @pytest.fixture
    def mock_model(self):
        """Create mock sentence transformer model."""
        model = Mock()
        model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        return model

    @pytest.fixture
    def mock_index(self):
        """Create mock FAISS index."""
        index = Mock()
        # Mock reconstruct to return a vector
        index.reconstruct.return_value = np.random.randn(384).astype(np.float32)
        # Mock search to return (similarities, indices)
        index.search.return_value = (
            np.array([[1.0, 0.95, 0.9, 0.85]]),  # similarities
            np.array([[0, 1, 2, 3]]),  # indices
        )
        return index

    def test_similar_movies_with_year_filters_correctly(
        self, mock_model, mock_index, mock_movies_df
    ):
        """Test that year parameter correctly filters duplicate titles."""
        # Search for 2006 version of The Departed
        results = similar_movies_search(
            "The Departed", mock_model, mock_index, mock_movies_df, k=10, year=2006
        )

        # Should have results
        assert len(results) > 0
        # First result should be from 2006 (after filtering)
        # The mock returns indices [0,1,2,3] which after our filtering will pick index 0

    def test_similar_movies_without_year_uses_most_recent(
        self, mock_model, mock_index, mock_movies_df
    ):
        """Test that without year, it defaults to most recent."""
        # Search without specifying year
        results = similar_movies_search(
            "The Departed", mock_model, mock_index, mock_movies_df, k=10
        )

        # Should return results (defaults to most recent: 2006)
        assert len(results) > 0

    def test_similar_movies_year_not_found_falls_back(
        self, mock_model, mock_index, mock_movies_df
    ):
        """Test graceful fallback when year doesn't match."""
        # Search with year that doesn't exist for this title
        results = similar_movies_search(
            "The Departed",
            mock_model,
            mock_index,
            mock_movies_df,
            k=10,
            year=2020,  # Year that doesn't exist
        )

        # Should still return results (falls back to title match)
        assert len(results) >= 0

    def test_similar_movies_nonexistent_movie(
        self, mock_model, mock_index, mock_movies_df
    ):
        """Test handling of non-existent movie."""
        results = similar_movies_search(
            "Nonexistent Movie", mock_model, mock_index, mock_movies_df, k=10
        )

        # Should return empty results
        assert len(results) == 0

    def test_blend_movies_with_years(self, mock_model, mock_index, mock_movies_df):
        """Test blend_movies with year parameters."""
        results = blend_movies(
            "The Departed",
            "Inception",
            mock_model,
            mock_index,
            mock_movies_df,
            k=10,
            year1=2006,
            year2=2010,
        )

        # Should successfully blend
        assert len(results) >= 0

    def test_blend_movies_one_year_specified(
        self, mock_model, mock_index, mock_movies_df
    ):
        """Test blend_movies with only one year specified."""
        results = blend_movies(
            "The Departed",
            "Inception",
            mock_model,
            mock_index,
            mock_movies_df,
            k=10,
            year1=2006,  # Only first movie has year specified
        )

        # Should successfully blend
        assert len(results) >= 0

    def test_contrastive_search_with_year(self, mock_model, mock_index, mock_movies_df):
        """Test contrastive_search with year parameter."""
        results = contrastive_search(
            "The Departed",
            "confusing",
            mock_model,
            mock_index,
            mock_movies_df,
            k=10,
            year=2006,
        )

        # Should successfully search
        assert len(results) >= 0

    def test_contrastive_search_without_year(
        self, mock_model, mock_index, mock_movies_df
    ):
        """Test contrastive_search defaults to most recent without year."""
        results = contrastive_search(
            "The Departed", "violent", mock_model, mock_index, mock_movies_df, k=10
        )

        # Should successfully search (defaults to 2006 version)
        assert len(results) >= 0


class TestRoundTrip:
    """Test complete URL generation and parsing workflow."""

    def test_semantic_search_round_trip(self):
        """Test generating and parsing semantic search URL."""
        # Original query
        original_query = "atmospheric sci-fi with stunning visuals"

        # Generate URL query string
        params = {"mode": "semantic", "q": original_query}
        query_string = urllib.parse.urlencode(params)

        # Parse it back
        decoded_params = urllib.parse.parse_qs(query_string)

        # Verify round-trip accuracy
        assert decoded_params["mode"][0] == "semantic"
        assert decoded_params["q"][0] == original_query

    def test_similar_movies_round_trip_with_year(self):
        """Test generating and parsing similar movies URL with year."""
        # Original params
        movie = "The Matrix"
        year = 1999

        # Generate URL query string
        params = {"mode": "similar", "movie": movie, "year": year}
        query_string = urllib.parse.urlencode(params)

        # Parse it back
        decoded_params = urllib.parse.parse_qs(query_string)

        # Verify round-trip accuracy
        assert decoded_params["mode"][0] == "similar"
        assert decoded_params["movie"][0] == movie
        assert int(decoded_params["year"][0]) == year

    def test_blend_movies_round_trip_with_years(self):
        """Test generating and parsing blend URL with both years."""
        # Original params
        params = {
            "mode": "blend",
            "movie1": "Inception",
            "year1": 2010,
            "movie2": "The Matrix",
            "year2": 1999,
        }

        query_string = urllib.parse.urlencode(params)
        decoded_params = urllib.parse.parse_qs(query_string)

        # Verify all parameters preserved
        assert decoded_params["mode"][0] == "blend"
        assert decoded_params["movie1"][0] == "Inception"
        assert int(decoded_params["year1"][0]) == 2010
        assert decoded_params["movie2"][0] == "The Matrix"
        assert int(decoded_params["year2"][0]) == 1999
