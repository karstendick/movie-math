# Movie Math V2 Features

This document contains potential features and improvements to implement after the MVP is complete.

---

## V2 Features

### Accessibility Improvements
**Priority: High** (moved to top of V2 list per user request)

- Alt text for movie posters
- Keyboard navigation support
- Screen reader compatibility
- ARIA labels for interactive elements
- High contrast mode support

---

### Testing & Quality

- Unit tests for data cleaning functions
- Unit tests for search functions with known inputs/outputs
- Integration tests for end-to-end search
- Edge case testing (empty queries, non-existent movies)
- Extend CI/CD to run pytest
- Code coverage reporting

---

### Mood Picker Tab

Pre-defined mood searches:
- "Need a Good Cry" → emotional dramas about loss and redemption
- "Brain Food" → thought-provoking films with complex themes
- "Pure Escapism" → fun adventure films with spectacular visuals
- "Cozy Evening" → heartwarming feel-good stories
- "Date Night" → romantic comedies with chemistry
- "Sunday Morning" → light wholesome family-friendly films

---

### Additional UI Features

- Search history/recent searches (session state)
- Watchlist/favorites (session state)
- "More like this" button on each movie card
- **Movie detail improvements:**
  - Link each movie to its TMDB page
  - Show all indexed information (top 5 actors, full description, genres, etc.)
  - Tighter spacing on movie cards to fit more results on screen
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
- **Search stats display:**
  - Show search metadata after each query (e.g., "Searched 20,000 movies from 140 years in 0.4 seconds")
  - Demonstrates the speed and scale of the RAG system
  - Could include: total movies indexed, date range of movies, query time, number of results returned

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

---

## Notes

- These features are not required for the MVP
- Focus on core RAG functionality first
- Accessibility should be prioritized when moving to V2
- **IMPORTANT: Implement testing infrastructure before adding new features** to prevent regressions
  - See Testing & Quality section above for specific testing goals
  - Consider this a prerequisite for V2 development
