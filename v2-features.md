# Movie Math V2 Features

This document contains potential features and improvements to implement after the MVP is complete.

---

### Additional UI Features

- Search history/recent searches (session state)
- Watchlist/favorites (session state)
- "More like this" button on each movie card
- **Movie detail improvements:**
  - ✅ Link each movie to its TMDB page
  - ✅ Show all indexed information (top 5 actors, full description, genres, etc.)
  - ✅ Tighter spacing on movie cards to fit more results on screen
- Expand/collapse movie details accordion

---

### Enhanced Search Features

- Similar movies graph visualization

---

### RAG Visualization & Learning Features
**Goal: Showcase RAG understanding and portfolio value**

- **Embedding space visualization:**
  - 2D/3D projection of movie embeddings (UMAP/t-SNE)
  - Interactive scatter plot where you can hover/click movies
  - Color-code by genre, decade, or rating
  - Show query vector and nearest neighbors in the space

- **Index inspection tools:**
  - View raw embedding vectors for any movie
  - Compare embedding similarity between any two movies
  - Show which text fields contributed most to a movie's embedding
  - Display FAISS index statistics (size, dimensions, number of vectors)

- **Search explainability:**
  - Visualize cosine similarity scores for top results
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
- Social sharing (generate shareable links)

---

### Data Management

- Scheduled data refresh (cron job + `setup.py --refresh`)

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
