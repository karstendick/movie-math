"""
Embeddings module for Movie Math.

Handles loading embedding models, generating embeddings, building FAISS index,
and saving/loading the search system.

Note: Heavy dependencies (faiss, sentence_transformers) are imported lazily
within functions to avoid slow startup times when not needed.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

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

# Configurable model - easy to experiment with different options
MODEL_NAME = "all-MiniLM-L6-v2"  # Default: fast and good quality

# Alternative options:
# MODEL_NAME = "all-mpnet-base-v2"  # Higher quality, 2x slower, 2x storage
#   - Best for: Portfolio/demo quality
#   - Tradeoff: 20 min setup vs 10 min, 60MB embeddings vs 30MB
# MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"  # Optimized for queries
#   - Best for: Query-document matching
#   - Tradeoff: Movie blending might be slightly worse


def load_embedding_model(model_name: str = MODEL_NAME) -> "SentenceTransformer":
    """
    Load Sentence Transformer model.

    Args:
        model_name: Model to load (default: from MODEL_NAME constant)

    Returns:
        SentenceTransformer model

    Model will be cached in ~/.cache/torch/sentence_transformers/
    First download may take a minute
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded successfully (dimension: {embedding_dim})")
    return model


def generate_embeddings(
    texts: List[str], model: "SentenceTransformer", batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings for list of texts.

    Args:
        texts: List of strings to embed
        model: SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        numpy array of shape (n_texts, embedding_dim)
        Embeddings are L2-normalized for cosine similarity
    """
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    logger.info(f"Batch size: {batch_size}")

    # Generate embeddings with progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )

    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    logger.info("Embeddings are L2-normalized for cosine similarity")

    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """
    Build FAISS index for fast similarity search.

    Use IndexFlatIP (Inner Product) for cosine similarity:
    - Requires L2-normalized embeddings (done in generate_embeddings)
    - Inner product of normalized vectors = cosine similarity
    - Better for sentence embeddings than Euclidean distance (L2)

    Dataset is small enough (~10-15K movies) for exact search.

    Args:
        embeddings: numpy array of shape (n_movies, embedding_dim)
        - Must be L2-normalized
        - 384 dims for MiniLM models
        - 768 dims for MPNet models

    Returns:
        FAISS IndexFlatIP
    """
    import faiss

    logger.info("Building FAISS index...")

    # Get embedding dimension
    n_movies, embedding_dim = embeddings.shape
    logger.info(f"Index for {n_movies} movies with {embedding_dim} dimensions")

    # Create IndexFlatIP for cosine similarity with normalized vectors
    index = faiss.IndexFlatIP(embedding_dim)

    # Add embeddings to index
    index.add(embeddings.astype(np.float32))

    logger.info(f"FAISS index built successfully ({index.ntotal} vectors)")

    return index


def save_index_and_embeddings(
    index: "faiss.IndexFlatIP",
    embeddings: np.ndarray,
    movies_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save FAISS index, embeddings, and movie metadata to disk.

    Files to create:
    - faiss_index.bin
    - embeddings.npy
    - movie_metadata.parquet

    Args:
        index: FAISS index
        embeddings: numpy array of embeddings
        movies_df: DataFrame with movie metadata
        output_dir: Directory to save files
    """
    import faiss

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    index_path = output_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")

    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path}")

    # Save movie metadata
    metadata_path = output_dir / "movie_metadata.parquet"
    movies_df.to_parquet(metadata_path, index=False)
    logger.info(f"Saved movie metadata to {metadata_path}")

    # Calculate and log file sizes
    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    embeddings_size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    metadata_size_mb = metadata_path.stat().st_size / (1024 * 1024)
    total_size_mb = index_size_mb + embeddings_size_mb + metadata_size_mb

    logger.info("File sizes:")
    logger.info(f"  - FAISS index: {index_size_mb:.1f} MB")
    logger.info(f"  - Embeddings: {embeddings_size_mb:.1f} MB")
    logger.info(f"  - Metadata: {metadata_size_mb:.1f} MB")
    logger.info(f"  - Total: {total_size_mb:.1f} MB")


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
    import faiss

    logger.info(f"Loading search system from {index_dir}")

    # Check that all required files exist
    index_path = index_dir / "faiss_index.bin"
    metadata_path = index_dir / "movie_metadata.parquet"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Movie metadata not found: {metadata_path}")

    # Load model
    model = load_embedding_model()

    # Load FAISS index
    index = faiss.read_index(str(index_path))
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")

    # Load movie metadata
    movies_df = pd.read_parquet(metadata_path)
    logger.info(f"Loaded metadata for {len(movies_df)} movies")

    logger.info("Search system loaded successfully")

    return model, index, movies_df
