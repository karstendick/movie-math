"""
Streamlit UI for Movie Math - Semantic Movie Search.

A web interface for searching movies using semantic search, contrastive search,
and movie blending powered by RAG (Retrieval-Augmented Generation).
"""

import logging
import time
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from search import (
    blend_movies,
    contrastive_search,
    load_search_system,
    semantic_search,
    similar_movies_search,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Movie Math - Semantic Movie Search",
        page_icon="\U0001f3ac",  # Film clapper emoji
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_custom_css():
    """Add custom CSS for movie cards and styling."""
    st.markdown(
        """
        <style>
        /* Main container styling - reduce padding */
        .main {
            padding: 0.5rem 1rem;
        }

        /* Reduce spacing in main block */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }

        /* Reduce header spacing */
        h1, h2, h3 {
            margin-top: 0rem;
            margin-bottom: 0.75rem;
        }

        /* Reduce tab content padding */
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }

        /* Movie card styling */
        .movie-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 8px;
            margin: 6px 0;
            transition: transform 0.2s, box-shadow 0.2s;
            background-color: white;
            height: 100%;
        }

        .movie-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        /* Poster image styling */
        .movie-poster {
            width: 100%;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 6px;
        }

        /* Movie title styling */
        .movie-title {
            font-weight: bold;
            font-size: 13px;
            margin-bottom: 3px;
            color: #1f1f1f;
            line-height: 1.2;
        }

        /* Movie info styling */
        .movie-info {
            font-size: 11px;
            color: #666;
            margin-bottom: 1px;
        }

        /* Similarity badge */
        .similarity-badge {
            display: inline-block;
            background-color: #4CAF50;
            color: white;
            padding: 3px 6px;
            border-radius: 10px;
            font-size: 10px;
            font-weight: bold;
            margin-bottom: 4px;
        }

        /* Rating badge */
        .rating-badge {
            color: #ff9800;
            font-weight: bold;
        }

        /* Genre tags */
        .genre-tag {
            display: inline-block;
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 1px 6px;
            border-radius: 8px;
            font-size: 10px;
            margin: 1px;
        }

        /* Footer styling */
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 12px;
        }

        /* Example button styling */
        .stButton button {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px 12px;
            font-size: 13px;
        }

        .stButton button:hover {
            background-color: #404040;
            border-color: #777;
        }

        /* Accessible input styling - keep focus indicators visible */
        div[data-baseweb="input"],
        div[data-baseweb="base-input"],
        input {
            border-color: #4a4a4a !important;
        }

        /* Enhanced focus styles for keyboard navigation accessibility */
        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="base-input"]:focus-within,
        input:focus {
            border-color: #0066cc !important;
            box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2) !important;
            outline: 2px solid #0066cc;
            outline-offset: 2px;
        }

        /* Accessible button focus styles */
        .stButton button:focus {
            outline: 2px solid #0066cc;
            outline-offset: 2px;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 14px;
            padding: 8px 16px;
        }

        /* Style radio buttons to look like tabs */
        /* Container for radio group */
        div[role="radiogroup"][data-baseweb="radio-group"] {
            gap: 0px;
            background-color: transparent;
            border-bottom: 2px solid #e0e0e0;
            padding: 0;
        }

        /* Individual radio button containers */
        div[role="radiogroup"] > label {
            background-color: transparent !important;
            border: none !important;
            border-bottom: 3px solid transparent !important;
            padding: 12px 24px !important;
            margin: 0 !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #666 !important;
            cursor: pointer !important;
            transition: all 0.2s !important;
            border-radius: 0 !important;
        }

        /* Hover state */
        div[role="radiogroup"] > label:hover {
            background-color: rgba(128, 128, 128, 0.15) !important;
        }

        /* Active/checked state - target the label when input is checked */
        div[role="radiogroup"] > label:has(input:checked) {
            border-bottom: 3px solid #1f77b4 !important;
            color: #1f1f1f !important;
            background-color: transparent !important;
        }

        /* Hide the radio button circles */
        div[role="radiogroup"] input[type="radio"] {
            display: none !important;
        }

        /* Hide the radio indicator */
        div[role="radiogroup"] label > div[data-testid="stMarkdownContainer"] {
            margin: 0 !important;
        }

        div[role="radiogroup"] label > div:first-child:not(
            [data-testid="stMarkdownContainer"]
        ) {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_system():
    """
    Load search system with caching.

    Error handling:
    - Check if required index files exist
    - Show error message and stop if files missing

    Returns:
        (model, index, movies_df)
    """
    index_dir = Path("data/index")

    # Check if required files exist
    required_files = [
        index_dir / "faiss_index.bin",
        index_dir / "embeddings.npy",
        index_dir / "movie_metadata.parquet",
    ]

    missing_files = [f for f in required_files if not f.exists()]

    if missing_files:
        st.error(
            "**Index files not found!**\n\n"
            "Please run the setup script first:\n\n"
            "```bash\n"
            "python setup.py\n"
            "```\n\n"
            f"Missing files: {', '.join(str(f) for f in missing_files)}"
        )
        st.stop()

    logger.info("Loading search system...")
    model, index, movies_df = load_search_system(index_dir)
    logger.info(f"Loaded {len(movies_df)} movies")

    return model, index, movies_df


def get_movie_autocomplete_options(movies_df: pd.DataFrame) -> List[str]:
    """
    Get list of movie titles for autocomplete, formatted with year.

    Args:
        movies_df: DataFrame with movie metadata

    Returns:
        List of formatted movie titles: "Title (Year)"
    """
    # Create formatted titles: "Movie Title (Year)"
    if "year" in movies_df.columns:
        options = [
            (
                f"{row['title']} ({int(row['year'])})"
                if pd.notna(row["year"])
                else row["title"]
            )
            for _, row in movies_df.iterrows()
        ]
    else:
        options = movies_df["title"].tolist()

    return sorted(options)


def parse_movie_title_from_selection(selection: str) -> str:
    """
    Extract movie title from autocomplete selection.

    Args:
        selection: Formatted string like "Inception (2010)"

    Returns:
        Just the title: "Inception"
    """
    # Remove year in parentheses if present
    if " (" in selection and selection.endswith(")"):
        return selection.rsplit(" (", 1)[0]
    return selection


def display_movie_grid(
    movies: pd.DataFrame,
    model,
    index,
    movies_df: pd.DataFrame,
    cols: int = 5,
    show_similarity: bool = True,
):
    """
    Display movies in a scrollable grid.

    Each movie card shows:
    - Poster image
    - Title and year
    - Similarity score (if show_similarity=True)
    - Rating
    - Genres
    - Director

    Args:
        movies: DataFrame with movie results
        cols: Number of columns in grid (default 5)
        show_similarity: Whether to show similarity scores (default True)
    """
    if len(movies) == 0:
        st.warning("No movies found matching your criteria.")
        return

    # Display movies in grid
    rows = (len(movies) + cols - 1) // cols

    for row in range(rows):
        row_cols = st.columns(cols)

        for col_idx in range(cols):
            movie_idx = row * cols + col_idx

            if movie_idx >= len(movies):
                break

            movie = movies.iloc[movie_idx]

            with row_cols[col_idx]:
                # Get movie details for accessibility
                title = movie["title"]
                year = movie.get("year", "")
                year_str = f" ({int(year)})" if pd.notna(year) else ""

                # Get TMDB URL
                tmdb_url = ""
                if "id" in movie and pd.notna(movie["id"]):
                    tmdb_url = f"https://www.themoviedb.org/movie/{int(movie['id'])}"

                # Poster with accessible description - make it clickable
                poster_url = movie.get("poster_url", "")
                if poster_url and pd.notna(poster_url):
                    # Use HTML with alt text for accessibility
                    if tmdb_url:
                        st.markdown(
                            f'<a href="{tmdb_url}" target="_blank">'
                            f'<img src="{poster_url}" '
                            f'alt="Movie poster for {title}{year_str}" '
                            f'style="width: 100%; border-radius: 6px; '
                            f"box-shadow: 0 2px 8px rgba(0,0,0,0.1); "
                            f'transition: opacity 0.2s;" '
                            f'onmouseover="this.style.opacity=0.8" '
                            f'onmouseout="this.style.opacity=1" /></a>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<img src="{poster_url}" '
                            f'alt="Movie poster for {title}{year_str}" '
                            f'style="width: 100%; border-radius: 6px; '
                            f'box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />',
                            unsafe_allow_html=True,
                        )
                else:
                    # Placeholder if no poster with accessible label
                    st.markdown(
                        f'<div role="img" aria-label="No poster available for {title}" '
                        f'style="background-color: #f0f0f0; height: 300px; '
                        "display: flex; align-items: center; justify-content: center; "
                        'border-radius: 6px;">\U0001f3ac</div>',
                        unsafe_allow_html=True,
                    )

                # Title with match % and rating - make it clickable
                if show_similarity and "similarity" in movie:
                    similarity = movie["similarity"]
                    # Convert cosine similarity to percentage
                    percentage = (similarity + 1) / 2 * 100
                    match_style = (
                        "color: #4CAF50; font-weight: bold; "
                        "font-size: 11px; margin-left: 6px;"
                    )
                    match_badge = (
                        f'<span style="{match_style}">'
                        f"{percentage:.0f}% match</span>"
                    )
                else:
                    match_badge = ""

                # Add rating badge
                if "rating" in movie and pd.notna(movie["rating"]):
                    rating = movie["rating"]
                    rating_style = "font-size: 11px; margin-left: 6px;"
                    rating_badge = (
                        f'<span style="{rating_style}" '
                        f'aria-label="Rating: {rating:.1f} out of 10">'
                        f"\u2b50 {rating:.1f}</span>"
                    )
                else:
                    rating_badge = ""

                if tmdb_url:
                    if pd.notna(year):
                        title_html = (
                            f'<a href="{tmdb_url}" target="_blank" '
                            f'style="text-decoration: none; '
                            f'color: inherit;">'
                            f'<strong style="font-size: 13px;">'
                            f"{title}</strong> "
                            f'<span style="font-size: 12px;">'
                            f"({int(year)})</span></a>"
                            f"{match_badge}{rating_badge}"
                        )
                        st.markdown(title_html, unsafe_allow_html=True)
                    else:
                        title_html = (
                            f'<a href="{tmdb_url}" target="_blank" '
                            f'style="text-decoration: none; '
                            f'color: inherit;">'
                            f'<strong style="font-size: 13px;">'
                            f"{title}</strong></a>"
                            f"{match_badge}{rating_badge}"
                        )
                        st.markdown(title_html, unsafe_allow_html=True)
                else:
                    if pd.notna(year):
                        title_html = (
                            f'<strong style="font-size: 13px;">'
                            f"{title}</strong> "
                            f'<span style="font-size: 12px;">'
                            f"({int(year)})</span>"
                            f"{match_badge}{rating_badge}"
                        )
                        st.markdown(title_html, unsafe_allow_html=True)
                    else:
                        title_html = (
                            f'<strong style="font-size: 13px;">'
                            f"{title}</strong>"
                            f"{match_badge}{rating_badge}"
                        )
                        st.markdown(title_html, unsafe_allow_html=True)

                # Genres
                if "genres" in movie:
                    genres = movie["genres"]
                    if isinstance(genres, list) and len(genres) > 0:
                        genres_str = ", ".join(genres)  # Show all
                        genre_style = (
                            "font-size: 11px; font-style: italic; "
                            "margin-top: 2px; margin-bottom: 2px;"
                        )
                        st.markdown(
                            f'<div style="{genre_style}">' f"{genres_str}</div>",
                            unsafe_allow_html=True,
                        )

                # Overview/Description
                if "overview" in movie:
                    overview = movie.get("overview", "")
                    if overview and not (
                        isinstance(overview, float) and pd.isna(overview)
                    ):
                        overview_style = (
                            "font-size: 10px; color: #555; "
                            "margin-top: 3px; margin-bottom: 3px; "
                            "line-height: 1.3;"
                        )
                        st.markdown(
                            f'<div style="{overview_style}">' f"{overview}</div>",
                            unsafe_allow_html=True,
                        )

                # Director
                if "director" in movie and pd.notna(movie["director"]):
                    director = movie["director"]
                    dir_style = (
                        "font-size: 11px; color: #666; "
                        "margin-top: 2px; margin-bottom: 1px;"
                    )
                    st.markdown(
                        f'<div style="{dir_style}">' f"Dir: {director}</div>",
                        unsafe_allow_html=True,
                    )

                # Cast (top 5 actors)
                if "cast" in movie:
                    cast = movie.get("cast")
                    has_len = hasattr(cast, "__len__")
                    if cast is not None and has_len and len(cast) > 0:
                        # Convert to list if it's a numpy array
                        to_list = hasattr(cast, "tolist")
                        cast_list = cast.tolist() if to_list else cast
                        cast_str = ", ".join(cast_list[:5])
                        cast_style = (
                            "font-size: 11px; color: #666; "
                            "margin-top: 1px; margin-bottom: 2px;"
                        )
                        st.markdown(
                            f'<div style="{cast_style}">' f"Cast: {cast_str}</div>",
                            unsafe_allow_html=True,
                        )

                # "More like this" button
                button_key = f"similar_{movie_idx}_{title.replace(' ', '_')}"
                if st.button(
                    "Find similar →",
                    key=button_key,
                    help=f"Find movies similar to {title}",
                    use_container_width=True,
                ):
                    # Store the movie for the Similar Movies tab
                    st.session_state.similar_movies_target = title
                    st.session_state.similar_movies_trigger = True
                    # Trigger the search
                    start_time = time.time()
                    results = similar_movies_search(
                        title, model, index, movies_df, k=100
                    )
                    search_time = time.time() - start_time
                    st.session_state.similar_movies_results = results
                    st.session_state.similar_movies_time = search_time
                    # Set flag to switch to Similar Movies tab
                    st.session_state.switch_to_similar = True
                    st.rerun()

                # Add minimal spacing
                st.markdown(
                    '<div style="margin-bottom: 2px;"></div>', unsafe_allow_html=True
                )


def display_footer():
    """Display footer with TMDb attribution."""
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
        This product uses the TMDb API but is not endorsed or certified
        by TMDb.<br>
        Movie data and posters provided by
        <a href="https://www.themoviedb.org" target="_blank">
        The Movie Database (TMDb)</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def tab_semantic_search(model, index, movies_df):
    """
    Tab 1: Basic semantic search.

    Features:
    - Text input for query
    - Example queries as buttons
    - Result grid
    """
    # Example queries
    example_cols = st.columns(4)
    examples = [
        "mind-bending thrillers with plot twists",
        "heartwarming films about family",
        "atmospheric sci-fi with stunning visuals",
        "dark comedies with clever dialogue",
    ]

    # Initialize session state
    if "semantic_results" not in st.session_state:
        st.session_state.semantic_results = None
    if "semantic_query_input" not in st.session_state:
        st.session_state.semantic_query_input = ""

    # Example buttons - execute search immediately when clicked
    for idx, example in enumerate(examples):
        with example_cols[idx]:
            if st.button(example, key=f"example_{idx}", width="stretch"):
                # Update query input and execute search
                st.session_state.semantic_query_input = example
                with st.spinner("Searching for movies..."):
                    start_time = time.time()
                    st.session_state.semantic_results = semantic_search(
                        example, model, index, movies_df, k=100
                    )
                    search_time = time.time() - start_time
                    st.session_state.semantic_search_time = search_time

    # Search form (allows Enter key submission)
    with st.form(key="semantic_search_form"):
        col1, col2 = st.columns([6, 1], gap="small")

        with col1:
            query = st.text_input(
                "Search for movies by theme, mood, or description",
                key="semantic_query_input",
                placeholder="e.g., movies about the cost of ambition",
                help=(
                    "Describe the type of movie you're looking for "
                    "using any words or phrases"
                ),
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Align button with input
            search_submitted = st.form_submit_button(
                "Search", use_container_width=True, type="primary"
            )

    # Execute search if form submitted
    if search_submitted and query and len(query.strip()) > 0:
        with st.spinner("Searching for movies..."):
            start_time = time.time()
            st.session_state.semantic_results = semantic_search(
                query, model, index, movies_df, k=100
            )
            search_time = time.time() - start_time
            st.session_state.semantic_search_time = search_time

    # Display results if they exist
    if st.session_state.semantic_results is not None:
        # Display search stats
        if "semantic_search_time" in st.session_state:
            num_movies = len(movies_df)
            search_time = st.session_state.semantic_search_time
            st.markdown(
                f'<div style="color: #666; font-size: 13px; margin-bottom: 1rem;">'
                f"\U0001f4ca Searched {num_movies:,} movies in {search_time:.2f}s"
                f"</div>",
                unsafe_allow_html=True,
            )

        display_movie_grid(
            st.session_state.semantic_results,
            model,
            index,
            movies_df,
            cols=5,
            show_similarity=True,
        )


def tab_contrastive_search(model, index, movies_df):
    """
    Tab 2: Like X But Not Y.

    Features:
    - Text input with autocomplete for movie selection
    - Text input for "avoid" aspects
    - Result grid
    """
    # Get autocomplete options
    movie_options = get_movie_autocomplete_options(movies_df)

    # Search form (allows Enter key submission)
    with st.form(key="contrastive_search_form"):
        # Movie selection with autocomplete
        selected_movie = st.selectbox(
            "Select a movie you like:",
            options=[""] + movie_options,
            index=0,
            placeholder="Search for a movie...",
            help="Start typing to search for movies in the database",
        )

        # Avoid text input with search button aligned
        col1, col2 = st.columns([6, 1], gap="small")

        with col1:
            avoid_text = st.text_input(
                "What aspects do you want to avoid?",
                placeholder="e.g., confusing, violent, sad",
                help="Describe themes, tones, or elements you want to avoid",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Align button with input
            search_submitted = st.form_submit_button(
                "Search", use_container_width=True, type="primary"
            )

    if (
        search_submitted
        and selected_movie
        and avoid_text
        and len(avoid_text.strip()) > 0
    ):
        # Parse movie title from selection
        movie_title = parse_movie_title_from_selection(selected_movie)

        with st.spinner(
            f"Finding movies like '{movie_title}' but not '{avoid_text}'..."
        ):
            start_time = time.time()
            results = contrastive_search(
                movie_title, avoid_text, model, index, movies_df, k=100
            )
            search_time = time.time() - start_time

        if len(results) == 0:
            st.error(
                f"Movie '{movie_title}' not found in the database. "
                f"Please try another movie."
            )
        else:
            # Display search stats
            num_movies = len(movies_df)
            st.markdown(
                f'<div style="color: #666; font-size: 13px; margin-bottom: 1rem;">'
                f"\U0001f4ca Searched {num_movies:,} movies in {search_time:.2f}s"
                f"</div>",
                unsafe_allow_html=True,
            )
            display_movie_grid(
                results, model, index, movies_df, cols=5, show_similarity=True
            )


def tab_movie_blender(model, index, movies_df):
    """
    Tab 3: Blend two movies.

    Features:
    - Two text inputs with autocomplete for movie selection
    - Result grid
    - Explanation of blend
    """
    # Get autocomplete options
    movie_options = get_movie_autocomplete_options(movies_df)

    # Two movie selections
    col1, col2 = st.columns(2)

    with col1:
        movie1 = st.selectbox(
            "Select first movie:",
            options=[""] + movie_options,
            index=0,
            key="blend_movie1_select",
            placeholder="Search for a movie...",
            help="Choose the first movie to blend",
        )

    with col2:
        movie2 = st.selectbox(
            "Select second movie:",
            options=[""] + movie_options,
            index=0,
            key="blend_movie2_select",
            placeholder="Search for a movie...",
            help="Choose the second movie to blend",
        )

    # Automatically search when both movies are selected
    if movie1 and movie2:
        # Parse movie titles from selections
        movie1_title = parse_movie_title_from_selection(movie1)
        movie2_title = parse_movie_title_from_selection(movie2)

        with st.spinner(f"Blending '{movie1_title}' and '{movie2_title}'..."):
            start_time = time.time()
            results = blend_movies(
                movie1_title, movie2_title, model, index, movies_df, k=100
            )
            search_time = time.time() - start_time

        if len(results) == 0:
            st.error(
                "One or both movies not found in the database. " "Please try again."
            )
        else:
            # Display search stats
            num_movies = len(movies_df)
            st.markdown(
                f'<div style="color: #666; font-size: 13px; margin-bottom: 1rem;">'
                f"\U0001f4ca Searched {num_movies:,} movies in {search_time:.2f}s"
                f"</div>",
                unsafe_allow_html=True,
            )
            display_movie_grid(
                results, model, index, movies_df, cols=5, show_similarity=True
            )


def tab_similar_movies(model, index, movies_df):
    """
    Tab 4: Similar Movies.

    Features:
    - Shows movies similar to a selected movie
    - Populated when user clicks "More like this" button
    - Can also manually search for similar movies
    """
    # Get autocomplete options
    movie_options = get_movie_autocomplete_options(movies_df)

    # Check if we have a movie from "More like this" button
    # and find its index in the options list
    default_index = 0
    if "similar_movies_target" in st.session_state:
        target_movie = st.session_state.similar_movies_target
        # Find the matching option in the list
        # Match format is "Title (Year)"
        matching_options = [
            opt
            for opt in movie_options
            if opt.startswith(target_movie + " (") or opt == target_movie
        ]
        if matching_options:
            # Find index in the full list (add 1 for the empty string at index 0)
            default_index = movie_options.index(matching_options[0]) + 1

    # Movie selection dropdown (no form, auto-search on selection)
    # Use the target from session state to determine selection
    if "similar_movies_trigger" in st.session_state:
        # Fresh search triggered by button - use the target
        del st.session_state.similar_movies_trigger

    selected_movie = st.selectbox(
        "Find movies similar to:",
        options=[""] + movie_options,
        index=default_index,
        placeholder="Search for a movie...",
        help="Start typing to search for movies in the database",
    )

    # Automatically search when a movie is selected
    if selected_movie:
        movie_title = parse_movie_title_from_selection(selected_movie)

        with st.spinner(f"Finding movies similar to '{movie_title}'..."):
            start_time = time.time()
            results = similar_movies_search(movie_title, model, index, movies_df, k=100)
            search_time = time.time() - start_time
            st.session_state.similar_movies_results = results
            st.session_state.similar_movies_time = search_time

    # Display results if they exist
    if "similar_movies_results" in st.session_state:
        results = st.session_state.similar_movies_results

        if len(results) == 0:
            st.error("Movie not found in the database. Please try another movie.")
        else:
            # Display search stats
            num_movies = len(movies_df)
            search_time = st.session_state.similar_movies_time
            st.markdown(
                f'<div style="color: #666; font-size: 13px; margin-bottom: 1rem;">'
                f"\U0001f4ca Searched {num_movies:,} movies in {search_time:.2f}s"
                f"</div>",
                unsafe_allow_html=True,
            )
            display_movie_grid(
                results, model, index, movies_df, cols=5, show_similarity=True
            )


def main():
    """Main application entry point."""
    setup_page_config()
    inject_custom_css()

    # Load search system
    model, index, movies_df = load_system()

    # Compact header
    st.markdown("## \U0001f3ac Movie Math")

    # Initialize active tab in session state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "\U0001f50d Semantic Search"

    # Check if we need to switch to Similar Movies tab
    if st.session_state.get("switch_to_similar", False):
        st.session_state.active_tab = "\U0001f3af Similar Movies"
        st.session_state.switch_to_similar = False

    # Tab selector
    active_tab = st.radio(
        "Select search mode:",
        [
            "\U0001f50d Semantic Search",
            "\U0001f3af Similar Movies",
            "\U0001f3ac Movie Blender",
            "\U0001f4af Like X But Not Y",
        ],
        key="active_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    # Display the active tab content
    if active_tab == "\U0001f50d Semantic Search":
        tab_semantic_search(model, index, movies_df)
    elif active_tab == "\U0001f3af Similar Movies":
        tab_similar_movies(model, index, movies_df)
    elif active_tab == "\U0001f3ac Movie Blender":
        tab_movie_blender(model, index, movies_df)
    elif active_tab == "\U0001f4af Like X But Not Y":
        tab_contrastive_search(model, index, movies_df)

    # Footer
    display_footer()


if __name__ == "__main__":
    main()
