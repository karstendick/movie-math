#!/usr/bin/env python3
"""
Interactive demo for Movie Math search functions.

Try out semantic search, contrastive search, and movie blending
with an interactive menu-driven interface.

Usage:
    python search_demo.py
"""

from pathlib import Path

from src.search import (
    blend_movies,
    contrastive_search,
    load_search_system,
    semantic_search,
)


def print_results(results, title="Results"):
    """Print search results in a nice format."""
    print(f"\n{title}")
    print("=" * 80)
    if len(results) == 0:
        print("No results found.")
        return

    for i, row in results.head(10).iterrows():
        similarity_pct = row["similarity"] * 100
        print(f"{similarity_pct:5.1f}% - {row['title']} ({row.get('year', 'N/A')})")
        if "genres" in row and row["genres"] is not None and len(row["genres"]) > 0:
            genres = ", ".join(row["genres"][:3])  # First 3 genres
            print(f"       Genres: {genres}")
    print()


def main():
    print("Loading search system...")
    model, index, movies_df = load_search_system(Path("data/index"))
    print(f"✓ Loaded {len(movies_df):,} movies\n")

    while True:
        print("\n" + "=" * 80)
        print("MOVIE SEARCH TEST MENU")
        print("=" * 80)
        print("1. Semantic Search (search by description)")
        print("2. Contrastive Search (like X but not Y)")
        print("3. Movie Blending (combine two movies)")
        print("4. Run example queries")
        print("5. Exit")
        print()

        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            print("\n--- SEMANTIC SEARCH ---")
            query = input("Enter search query: ").strip()
            if query:
                results = semantic_search(query, model, index, movies_df, k=10)
                print_results(results, f'Results for: "{query}"')

        elif choice == "2":
            print("\n--- CONTRASTIVE SEARCH ---")
            movie = input("Like this movie: ").strip()
            avoid = input("But not (aspect to avoid): ").strip()
            if movie and avoid:
                results = contrastive_search(
                    movie, avoid, model, index, movies_df, k=10
                )
                print_results(results, f'Like "{movie}" but not "{avoid}"')

        elif choice == "3":
            print("\n--- MOVIE BLENDING ---")
            movie1 = input("First movie: ").strip()
            movie2 = input("Second movie: ").strip()
            if movie1 and movie2:
                results = blend_movies(movie1, movie2, model, index, movies_df, k=10)
                print_results(results, f'Blend: "{movie1}" + "{movie2}"')

        elif choice == "4":
            # Run example queries
            print("\n--- EXAMPLE QUERIES ---")

            # Example 1: Semantic
            query = "atmospheric sci-fi with stunning visuals"
            results = semantic_search(query, model, index, movies_df, k=5)
            print_results(results, f'Semantic: "{query}"')

            # Example 2: Contrastive
            results = contrastive_search(
                "The Godfather", "violent", model, index, movies_df, k=5
            )
            print_results(
                results, 'Contrastive: Like "The Godfather" but not "violent"'
            )

            # Example 3: Blend
            results = blend_movies(
                "Grand Budapest Hotel", "Knives Out", model, index, movies_df, k=5
            )
            print_results(results, 'Blend: "Grand Budapest Hotel" + "Knives Out"')

        elif choice == "5":
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid choice. Please select 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
