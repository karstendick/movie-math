# Movie Math V2 Features

This document contains potential features and improvements to implement after the MVP is complete.

---

### Additional UI Features

- ~~Search history/recent searches (session state)~~ - Out of scope
- ~~Watchlist/favorites (session state)~~ - Out of scope
- ✅ "More like this" button on each movie card
  - ✅ Auto-switches to Similar Movies tab
  - ✅ Pre-populates dropdown with selected movie
  - ✅ Uses existing embeddings from FAISS index (fast!)
- **Movie detail improvements:**
  - ✅ Link each movie to its TMDB page
  - ✅ Show all indexed information (top 5 actors, full description, genres, etc.)
  - ✅ Tighter spacing on movie cards to fit more results on screen
- ~~Expand/collapse movie details accordion~~ - Out of scope

---

### Enhanced Search Features

- Similar movies graph visualization

---

### RAG Visualization & Learning Features
**Goal: Showcase RAG understanding and portfolio value**

- **✅ Embedding space visualization:**
  - **Design decisions:**
    - ✅ Use **UMAP** for dimensionality reduction (384D → 2D)
      - Better preservation of both local and global structure vs t-SNE
      - Fast computation (~10-30s for 20k movies)
      - Use cosine metric to match FAISS similarity
    - ✅ **Precompute** 2D coordinates during setup
      - Save to `data/index/embeddings_2d.npy`
      - Make step idempotent: check if file exists before recomputing
      - Skip UMAP if embeddings_2d.npy already exists
      - Load instantly when app starts (no performance hit)
    - ✅ Use **Plotly** for interactive visualization
      - Hover tooltips showing movie title, year, genres
      - Zoom, pan, and selection capabilities
      - Streamlit integration via `st.plotly_chart()`
  - **UI Design:**
    - ✅ Add "📊 Visualize" button next to search stats (after Share button)
    - ✅ Button expands inline visualization below search stats
    - ✅ Shows all movies in the space with search results highlighted
    - ✅ Can collapse visualization to return to grid view
  - **Additional features to implement:**
    - ✅ Interactive scatter plot with hover tooltips
    - ✅ Highlight current search results in the visualization
    - Color-code by genre, decade, or rating
    - Toggle genres on/off for filtering
    - Click a movie in the plot to see similar movies

- **Index inspection tools:**
  - View raw embedding vectors for any movie
  - Compare embedding similarity between any two movies
  - Show which text fields contributed most to a movie's embedding
  - ✅ Display FAISS index statistics (size, dimensions, number of vectors)

- **Search explainability:**
  - ✅ Visualize cosine similarity scores for top results
  - Show the generated embedding for user queries
  - Highlight which query terms matched which movie attributes
  - Side-by-side comparison of query vector vs. result vectors

- **Debug/learning mode:**
  - Toggle to show technical details (embedding distances, retrieval times)
  - Export search results with similarity scores to JSON
  - "Behind the scenes" tab explaining how the RAG system works
  - Example queries with explanations of why results were returned

---

### External Integrations

- Streaming availability (JustWatch API)
- ✅ Social sharing (generate shareable links)

---

### Enhanced Result Explanations

- Matching attributes highlighting (which fields contributed to match)
- "Similar to these movies" context
- Keyword highlighting in descriptions
- **✅ Search stats display:**
  - ✅ Show search metadata after each query (e.g., "Searched 20,000 movies in 0.4 seconds")
  - ✅ Demonstrates the speed and scale of the RAG system
  - Could include: date range of movies, number of results returned

---

### Evaluation & Monitoring

- Automated evaluation metrics:
  - Precision@K with ground truth test queries
  - Query latency monitoring and optimization
  - A/B testing different embedding models

---

## Advanced RAG Techniques

For learning more advanced RAG concepts:

- **Hybrid search** (semantic + keyword)
- **Re-ranking** with cross-encoder
- **Query expansion**
- **Relevance feedback**
