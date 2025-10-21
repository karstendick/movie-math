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

- Random movie button ("Surprise me!")
- Search history/recent searches (session state)
- Watchlist/favorites (session state)
- "More like this" button on each movie card
- Export results to CSV/JSON

---

### Enhanced Search Features

- Configurable contrastive search weight (slider in UI from 0.0 to 1.0)
- Advanced search with metadata filters (genre, rating, year range)
- Similar movies graph visualization
- Director/actor filtering

---

### External Integrations

- Streaming availability (JustWatch API)
- Social sharing (generate shareable links)

---

### Data Management

- Scheduled data refresh (cron job + `setup.py --refresh`)
- `--skip-credits` flag for faster development setup

---

### Enhanced Result Explanations

- Matching attributes highlighting (which fields contributed to match)
- "Similar to these movies" context
- Keyword highlighting in descriptions

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
