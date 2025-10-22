"""
Movie Math setup script - Phases 1 & 2 implementation.

Idempotent setup that fetches and processes movie data from TMDb API,
then generates embeddings and builds FAISS index.

Usage:
    python setup.py                    # Run setup, skip if data/index exists
    python setup.py --redownload-data  # Re-fetch data and regenerate everything
    python setup.py --regenerate-index # Keep data, regenerate embeddings/index only
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_preparation import (
    add_poster_urls,
    clean_movies,
    load_processed_data,
    load_tmdb_data_from_api,
    prepare_search_text,
    save_processed_data,
)
from src.embeddings import (
    build_faiss_index,
    generate_embeddings,
    load_embedding_model,
    save_index_and_embeddings,
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def data_exists():
    """
    Check if processed data already exists.

    Returns:
        bool: True if data exists, False otherwise
    """
    processed_path = Path("data/processed/movies_clean.parquet")
    return processed_path.exists()


def index_exists():
    """
    Check if FAISS index and embeddings already exist.

    Returns:
        bool: True if index exists, False otherwise
    """
    index_path = Path("data/index/faiss_index.bin")
    embeddings_path = Path("data/index/embeddings.npy")
    metadata_path = Path("data/index/movie_metadata.parquet")
    return index_path.exists() and embeddings_path.exists() and metadata_path.exists()


def main():
    """Run Phase 1 & 2 setup (Data Preparation + Embeddings)."""
    parser = argparse.ArgumentParser(description="Movie Math Setup (Phases 1 & 2)")
    parser.add_argument(
        "--redownload-data",
        action="store_true",
        help="Re-fetch data from TMDb API and regenerate everything",
    )
    parser.add_argument(
        "--regenerate-index",
        action="store_true",
        help="Regenerate embeddings and index only (keep existing data)",
    )
    args = parser.parse_args()

    logger.info("Movie Math Setup - Phases 1 & 2")
    logger.info("=" * 60)

    # Get API key
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.error("TMDB_API_KEY not found in .env file")
        logger.error("Please add your TMDb API key to .env file")
        return

    # Determine what needs to be done
    if args.redownload_data:
        logger.info("Re-downloading data from TMDb API and regenerating everything...")
        fetch_data = True
        generate_index = True
    elif args.regenerate_index:
        logger.info("Regenerate index mode: Keeping data, regenerating index...")
        fetch_data = False
        generate_index = True
        # Verify data exists
        if not data_exists():
            logger.error("Processed data not found!")
            logger.error("Cannot regenerate index without data.")
            logger.error("Run without --regenerate-index to fetch data first.")
            return
    else:
        fetch_data = not data_exists()
        generate_index = not index_exists()

        if not fetch_data and not generate_index:
            logger.info("All files exist. Nothing to do!")
            logger.info("Use --redownload-data to re-fetch from TMDb API")
            logger.info("Use --regenerate-index to regenerate embeddings/index only")
            # Show existing data info
            try:
                df = load_processed_data(Path("data/processed/movies_clean.parquet"))
                logger.info(f"Existing data: {len(df)} movies")
                logger.info("Index location: data/index/")
                logger.info("Next: Implement Phase 3 (Search)")
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
            return

    # Fetch and process data
    if fetch_data:
        logger.info("Fetching movie data from TMDb API...")
        if args.redownload_data:
            logger.info("Re-download mode: Ignoring cache and re-fetching from API")
            logger.info("This will take approximately 70 minutes")
        else:
            logger.info("Cache will be used if available (much faster than full fetch)")

        try:
            # Fetch all movies that meet quality criteria
            logger.info("Step 1/5: Fetching all movies and credits from TMDb...")
            # If --redownload-data, don't use cache (use_cache=False)
            # Otherwise, use cache if available (use_cache=True)
            use_cache = not args.redownload_data
            movies_df = load_tmdb_data_from_api(api_key, use_cache=use_cache)
            logger.info(f"Fetched {len(movies_df)} movies")

            # Clean the data
            logger.info("Step 2/5: Cleaning movie data...")
            movies_df = clean_movies(movies_df)
            logger.info(f"After cleaning: {len(movies_df)} movies")

            # Add poster URLs
            logger.info("Step 3/5: Adding poster URLs...")
            movies_df = add_poster_urls(movies_df)

            # Prepare search text
            logger.info("Step 4/5: Preparing search text for embeddings...")
            movies_df = prepare_search_text(movies_df)

            # Save processed data
            output_path = Path("data/processed/movies_clean.parquet")
            logger.info(f"Step 5/5: Saving processed data to {output_path}...")
            save_processed_data(movies_df, output_path)

            # Display Phase 1 summary
            logger.info("=" * 60)
            logger.info("SUCCESS! Phase 1 complete.")
            logger.info("=" * 60)
            logger.info(f"Processed {len(movies_df)} movies")
            logger.info(f"Saved to: {output_path}")
            logger.info("Sample movies:")
            for idx, row in movies_df.head(3).iterrows():
                logger.info(f"  - {row['title']} ({int(row['year'])})")

        except ValueError as e:
            logger.error(f"ERROR: {e}")
            logger.error("Check your TMDb API key in .env file")
            return
        except Exception as e:
            logger.error(f"ERROR: {e}")
            logger.error("Something went wrong during data fetch")
            return
    else:
        # Load existing data
        logger.info("Loading existing processed data...")
        try:
            movies_df = load_processed_data(Path("data/processed/movies_clean.parquet"))
            logger.info(f"Loaded {len(movies_df)} movies from cache")
        except Exception as e:
            logger.error(f"ERROR: Failed to load processed data: {e}")
            logger.error("Try running with --force to regenerate")
            return

    # Phase 2: Generate embeddings and build index
    if generate_index:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Phase 2: Generating Embeddings and Building Index")
        logger.info("=" * 60)

        try:
            # Load embedding model
            logger.info("Step 1/4: Loading embedding model...")
            model = load_embedding_model()

            # Generate embeddings
            logger.info("Step 2/4: Generating embeddings...")
            logger.info("This may take 10-20 minutes depending on your CPU...")
            search_texts = movies_df["search_text"].tolist()
            embeddings = generate_embeddings(search_texts, model, batch_size=32)

            # Build FAISS index
            logger.info("Step 3/4: Building FAISS index...")
            index = build_faiss_index(embeddings)

            # Save everything
            logger.info("Step 4/4: Saving index and embeddings...")
            output_dir = Path("data/index")
            save_index_and_embeddings(index, embeddings, movies_df, output_dir)

            # Display Phase 2 summary
            logger.info("=" * 60)
            logger.info("SUCCESS! Phase 2 complete.")
            logger.info("=" * 60)
            logger.info(f"Indexed {len(movies_df)} movies")
            logger.info(f"Index saved to: {output_dir}")

        except Exception as e:
            logger.error(f"ERROR: {e}")
            logger.error("Something went wrong during embeddings generation")
            return
    else:
        logger.info("")
        logger.info("Skipping Phase 2 (index already exists)")
        logger.info("Use --force to regenerate")

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SETUP COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total movies indexed: {len(movies_df)}")
    logger.info("Next: Implement Phase 3 (Search)")
    logger.info("")
    logger.info("To run the app (once implemented):")
    logger.info("  streamlit run src/app.py")


if __name__ == "__main__":
    main()
