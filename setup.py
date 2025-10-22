"""
Movie Math setup script - Phase 1 implementation (Data Preparation only).

Idempotent setup that fetches and processes movie data from TMDb API.
Phase 2 (embeddings) and beyond will be added later.

Usage:
    python setup.py              # Run setup, skip if data exists
    python setup.py --force      # Force regenerate data
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


def main():
    """Run Phase 1 data preparation setup."""
    parser = argparse.ArgumentParser(description="Movie Math Setup (Phase 1)")
    parser.add_argument(
        "--force", action="store_true", help="Force regenerate all files"
    )
    args = parser.parse_args()

    logger.info("Movie Math Setup - Phase 1: Data Preparation")
    logger.info("=" * 60)

    # Get API key
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.error("TMDB_API_KEY not found in .env file")
        logger.error("Please add your TMDb API key to .env file")
        return

    # Determine what needs to be done
    if args.force:
        logger.info("Force mode: Regenerating data...")
        fetch_data = True
    else:
        fetch_data = not data_exists()

        if not fetch_data:
            logger.info("All files exist. Nothing to do!")
            logger.info("Use --force to regenerate data")
            # Show existing data info
            try:
                df = load_processed_data(Path("data/processed/movies_clean.parquet"))
                logger.info(f"Existing data: {len(df)} movies")
                logger.info("Next: Implement Phase 2 (Embeddings)")
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
            return

    # Fetch and process data
    if fetch_data:
        logger.info("Fetching movie data from TMDb API...")
        if args.force:
            logger.info("Force mode enabled: Ignoring cache and re-fetching from API")
            logger.info("This will take approximately 70 minutes")
        else:
            logger.info("Cache will be used if available (much faster than full fetch)")

        try:
            # Fetch all movies that meet quality criteria
            logger.info("Step 1/5: Fetching all movies and credits from TMDb...")
            # If --force, don't use cache (use_cache=False)
            # Otherwise, use cache if available (use_cache=True)
            use_cache = not args.force
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

            # Display summary
            logger.info("=" * 60)
            logger.info("SUCCESS! Phase 1 complete.")
            logger.info("=" * 60)
            logger.info(f"Processed {len(movies_df)} movies")
            logger.info(f"Saved to: {output_path}")
            logger.info("Sample movies:")
            for idx, row in movies_df.head(3).iterrows():
                logger.info(f"  - {row['title']} ({int(row['year'])})")
            logger.info("Next: Implement Phase 2 (Embeddings)")

        except ValueError as e:
            logger.error(f"ERROR: {e}")
            logger.error("Check your TMDb API key in .env file")
            return
        except Exception as e:
            logger.error(f"ERROR: {e}")
            logger.error("Something went wrong during data fetch")
            return


if __name__ == "__main__":
    main()
