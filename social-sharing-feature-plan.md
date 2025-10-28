# Social Sharing Feature Plan

## Overview
Enable users to share their Movie Math searches and results via shareable links. When someone opens a shared link, they should see the same search results and context as the original user.

---

## Use Cases

1. **Share Search Results**: User performs a semantic search and wants to share their findings
2. **Share Similar Movies**: User discovers interesting similar movies and wants to share the recommendation
3. **Portfolio/Demo**: Showcase specific impressive search results for portfolio purposes

---

## What Should Be Shareable?

### Core State to Encode:
- **Search type** (semantic search vs similar movies vs blend vs contrastive)
- **Query text** (for semantic search)
- **Selected movie** (for similar movies search)
- **Two movies** (for movie blender)
- **Movie + avoid text** (for contrastive search)

### Not Included (Out of Scope):
- **Number of results** - Always fixed at k=100, not user-configurable
- Scroll position to specific movie
- Expanded/collapsed state of movie cards

---

## Technical Approach

### **SELECTED: Option 1 - URL Query Parameters**

We're going with simple, human-readable URL query parameters for the initial implementation.

**Why this approach:**
- Simple to implement with Streamlit's native `st.query_params` API
- Human-readable URLs make debugging easy
- No backend/database required
- Easy to share and understand
- Movie queries are typically short enough to fit in URL limits

**Implementation:**
```
https://moviemath.com/?mode=semantic&q=movies+about+time+travel
https://moviemath.com/?mode=similar&movie=The+Matrix
https://moviemath.com/?mode=blend&movie1=Inception&movie2=The+Matrix
https://moviemath.com/?mode=contrastive&like=The+Godfather&avoid=violent
```

**URL Length Considerations:**
- Modern browsers support ~2000+ character URLs
- Movie titles: typically < 50 chars
- Search queries: typically < 200 chars
- Well within safe limits for our use case

---

### Alternative Options (Not Selected)

<details>
<summary>Option 2: Base64 Encoded State (click to expand)</summary>

**Pros:** Compact URLs, can encode complex state
**Cons:** Not human-readable, harder to debug
**Implementation:** `https://moviemath.com/?share=eyJtb2RlIjoic2VtYW50aWMi...`

</details>

<details>
<summary>Option 3: Short Links with Backend (click to expand)</summary>

**Pros:** Very short URLs, analytics, link management
**Cons:** Requires backend/database, more complex
**Implementation:** `https://moviemath.com/s/abc123`

**Note:** Could migrate to this later if URL length becomes an issue or we want analytics.

</details>

---

## Implementation Plan

### Phase 1: Basic URL Parameter Sharing

#### 1. URL State Management (Streamlit-specific)
- [ ] Add `st.query_params` handling in `main()` function
- [ ] Read URL query parameters on app load
- [ ] Create utility functions to serialize/deserialize app state
- [ ] Handle special characters and URL encoding

#### 2. State Restoration from URL
- [ ] Check for URL parameters at app start (before loading tabs)
- [ ] Restore `st.session_state` from URL parameters
- [ ] Set active tab based on `mode` parameter
- [ ] Auto-populate input fields and trigger search
- [ ] Handle invalid/missing parameters gracefully
- [ ] **Update search functions** to accept optional `year` parameter for disambiguation
  - Modify `similar_movies_search()` to filter by year if provided
  - Modify `blend_movies()` to filter by year if provided
  - Modify `contrastive_search()` to filter by year if provided

#### 3. Share Button UI Integration
- [ ] Add "Share" button next to search stats display
  - Location: After the search stats text (e.g., "Searched 20,000 movies in 0.4s")
  - Appears only when results exist
- [ ] Generate shareable URL from current session state
- [ ] Use Streamlit's clipboard API or custom HTML/JavaScript
- [ ] Show success message using `st.success()` or toast

#### 4. URL Parameter Schema (Streamlit implementation)
```python
# URL query parameter structure
{
  'mode': 'semantic' | 'similar' | 'blend' | 'contrastive',
  'q': str,              # query text (semantic mode)
  'movie': str,          # movie title (similar mode)
  'year': str,           # optional: movie year for disambiguation (similar mode)
  'movie1': str,         # first movie (blend mode)
  'year1': str,          # optional: first movie year (blend mode)
  'movie2': str,         # second movie (blend mode)
  'year2': str,          # optional: second movie year (blend mode)
  'like': str,           # movie to match (contrastive mode)
  'likeyear': str,       # optional: movie year for disambiguation (contrastive mode)
  'avoid': str,          # aspect to avoid (contrastive mode)
}
```

Example URLs:
- Semantic: `?mode=semantic&q=atmospheric+sci-fi+with+stunning+visuals`
- Similar: `?mode=similar&movie=The+Matrix&year=1999`
- Similar (no year): `?mode=similar&movie=The+Matrix` (defaults to most recent)
- Blend: `?mode=blend&movie1=Inception&year1=2010&movie2=The+Matrix&year2=1999`
- Contrastive: `?mode=contrastive&like=The+Godfather&likeyear=1972&avoid=violent`

---

## UI/UX Design

### Share Button Placement
**SELECTED: Next to search stats** ("Searched 20,000 movies in 0.4 seconds")

- Appears in the same row as the search stats
- Only visible when search results exist
- Compact and unobtrusive design
- Natural location - users see stats, then can share

### Share Flow
1. User clicks "Share" button
2. URL is generated from current state
3. URL is copied to clipboard automatically
4. Show success message: "✓ Link copied!" (inline, next to button)

### Visual Design
- Button style: Subtle, matches existing app design
- Icon: 🔗 (link icon)
- Tooltip: "Share these results"
- Success feedback: "✓ Link copied!" appears briefly next to button

---

## Edge Cases to Handle

1. **No Search Performed**: Hide share button until results exist
2. **Duplicate Movie Titles**: Multiple movies with same title (e.g., "The Departed" 2006 vs other versions)
   - **Solution**: Include year in URL for similar/blend/contrastive modes
   - URL format: `?mode=similar&movie=The+Departed&year=2006`
   - Search function should use both title AND year when year is provided
   - Falls back to "most recent" behavior if year not specified (current behavior)
3. **Long Query Strings**: Rare but possible; browser URL limits ~2000 chars (graceful handling if exceeded)
4. **Movie Not Found**: Shared link references movie no longer in index → show friendly error message
5. **Browser Compatibility**: Clipboard API not available in older browsers → fallback to displaying URL for manual copy
6. **Security**: Sanitize/validate URL parameters to prevent XSS attacks
7. **Special Characters**: Properly encode/decode movie titles with special characters (e.g., "Amélie", "10 Things I Hate About You")

---

## Testing Strategy

### Unit Tests
- URL serialization/deserialization
- State restoration logic
- Edge case handling

### Integration Tests
- Share button generates correct URL
- Opening shared URL restores state correctly
- Search auto-triggers with shared parameters

### Manual Testing
- Test various query types and lengths
- Test on different browsers
- Test mobile vs desktop
- Test with special characters in queries
- **Test duplicate movie titles** (e.g., share "The Departed" 2006, verify correct one loads)
- Test movies with and without years in database

---

## Success Criteria

- Share button appears correctly when results exist
- Shared URLs successfully restore search state
- Copy-to-clipboard works reliably
- All 4 search modes (semantic, similar, blend, contrastive) can be shared
- Special characters in movie titles/queries are handled correctly
- User feedback is positive

---

## Streamlit-Specific Implementation Details

### Query Parameters API
Streamlit provides native query parameter support via `st.query_params`:
```python
# Read query params
params = st.query_params
mode = params.get('mode', None)
query = params.get('q', None)

# Write query params
st.query_params['mode'] = 'semantic'
st.query_params['q'] = 'time travel movies'
```

### Share Button Implementation

**SELECTED: JavaScript with Custom HTML Component**

Using Streamlit's `components.html()` to create a custom share button with JavaScript clipboard functionality:

```python
import streamlit.components.v1 as components

def create_share_button(share_url):
    """Display share button with JavaScript clipboard functionality."""
    components.html(f"""
        <div style="margin: 10px 0;">
            <button onclick="copyLink()"
                    style="background: #4CAF50; color: white; border: none;
                           padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                🔗 Share Results
            </button>
            <span id="copy-feedback" style="margin-left: 10px; color: #4CAF50; display: none;">
                ✓ Link copied!
            </span>
        </div>
        <script>
        function copyLink() {{
            navigator.clipboard.writeText('{share_url}')
                .then(() => {{
                    document.getElementById('copy-feedback').style.display = 'inline';
                    setTimeout(() => {{
                        document.getElementById('copy-feedback').style.display = 'none';
                    }}, 2000);
                }})
                .catch(err => {{
                    // Fallback for older browsers
                    alert('Copy failed. URL: {share_url}');
                }});
        }}
        </script>
    """, height=50)
```

**Why this approach:**
- Native clipboard API support (works in modern browsers)
- Clean, inline feedback ("✓ Link copied!")
- No external dependencies needed
- Fallback alert for older browsers

### Session State Integration
Current session state structure (from app.py):
```python
# Semantic search
st.session_state.semantic_results
st.session_state.semantic_query_input
st.session_state.semantic_search_time

# Similar movies
st.session_state.similar_movies_target
st.session_state.similar_movies_results
st.session_state.similar_movies_time
st.session_state.switch_to_similar

# Active tab
st.session_state.active_tab
```

### Implementation Flow
1. User performs search → results stored in session state
2. User clicks "Share" → generate URL from session state
3. New user opens shared URL → params parsed into session state
4. App auto-triggers search based on restored state

## Technical Dependencies

### Dependencies
- **Streamlit** (native query params support via `st.query_params`)
- **Streamlit components** (`streamlit.components.v1` for custom HTML/JS)
- **Python urllib** (URL encoding/decoding)
- No external libraries needed - using native browser Clipboard API

---

## Design Decisions

1. **Encode results or just query?** → Just the query/parameters; re-run search on shared link open
   - Keeps URLs short and simple
   - Ensures users see up-to-date results if index changes

2. **Handle index changes?** → Yes, shared links re-run searches with current index
   - Results may differ if movies added/removed
   - This is acceptable and expected behavior

3. **User accounts/authentication?** → No, not needed
   - Sharing is anonymous
   - No tracking or analytics required

4. **Number of results?** → Fixed at k=100, not exposed to users

---

## Code Implementation Example

### 1. Add URL parameter restoration in `main()`:
```python
def main():
    """Main application entry point."""
    setup_page_config()
    inject_custom_css()

    # Load search system
    model, index, movies_df = load_system()

    # === NEW: Restore state from URL parameters ===
    restore_state_from_url(model, index, movies_df)
    # === END NEW ===

    # Compact header
    st.markdown("## 🎬 Movie Math")

    # ... rest of main() ...
```

### 2. Create `restore_state_from_url()` function:
```python
def restore_state_from_url(model, index, movies_df):
    """Restore app state from URL query parameters."""
    params = st.query_params

    mode = params.get('mode')
    if not mode:
        return  # No shared state in URL

    # Restore based on mode
    if mode == 'semantic':
        query = params.get('q')
        if query:
            st.session_state.active_tab = "🔍 Semantic Search"
            st.session_state.semantic_query_input = query
            # Auto-trigger search
            results = semantic_search(query, model, index, movies_df, k=100)
            st.session_state.semantic_results = results

    elif mode == 'similar':
        movie = params.get('movie')
        year = params.get('year')  # Optional year for disambiguation
        if movie:
            st.session_state.active_tab = "🎯 Similar Movies"
            st.session_state.similar_movies_target = movie
            # Auto-trigger search with optional year
            results = similar_movies_search(
                movie, model, index, movies_df, k=100, year=year
            )
            st.session_state.similar_movies_results = results

    # ... handle other modes (blend, contrastive) ...
    # Note: blend and contrastive modes will also need year handling
```

### 3. Add share button helper function:
```python
def generate_share_url(mode: str, **params) -> str:
    """Generate shareable URL from current state."""
    import urllib.parse

    base_url = "https://your-app-url.streamlit.app"  # Or st.get_option("browser.serverAddress")
    query_string = urllib.parse.urlencode({'mode': mode, **params})
    return f"{base_url}?{query_string}"

def display_share_button(share_url: str):
    """Display share button with copy-to-clipboard."""
    import streamlit.components.v1 as components

    components.html(f"""
        <div style="margin: 10px 0;">
            <button onclick="copyLink()"
                    style="background: #4CAF50; color: white; border: none;
                           padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                📋 Share Results
            </button>
            <span id="copy-feedback" style="margin-left: 10px; color: #4CAF50; display: none;">
                ✓ Link copied!
            </span>
        </div>
        <script>
        function copyLink() {{
            navigator.clipboard.writeText('{share_url}').then(() => {{
                document.getElementById('copy-feedback').style.display = 'inline';
                setTimeout(() => {{
                    document.getElementById('copy-feedback').style.display = 'none';
                }}, 2000);
            }});
        }}
        </script>
    """, height=50)
```

### 4. Integrate share button in each tab (example for semantic search):
```python
def tab_semantic_search(model, index, movies_df):
    # ... existing code ...

    # Display results if they exist
    if st.session_state.semantic_results is not None:
        # Display search stats
        if "semantic_search_time" in st.session_state:
            num_movies = len(movies_df)
            search_time = st.session_state.semantic_search_time

            # Create columns for stats and share button
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f'<div style="color: #666; font-size: 13px; margin-bottom: 1rem;">'
                    f"📊 Searched {num_movies:,} movies in {search_time:.2f}s"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col2:
                # === NEW: Share button ===
                query = st.session_state.semantic_query_input
                share_url = generate_share_url('semantic', q=query)
                display_share_button(share_url)
                # === END NEW ===

        # Display grid
        display_movie_grid(...)
```

### 5. Example for Similar Movies tab (with year extraction):
```python
def tab_similar_movies(model, index, movies_df):
    # ... existing code ...

    # Display results if they exist
    if "similar_movies_results" in st.session_state:
        results = st.session_state.similar_movies_results

        if len(results) > 0:
            # Create columns for stats and share button
            col1, col2 = st.columns([3, 1])

            with col1:
                # Display search stats
                st.markdown(...)

            with col2:
                # === NEW: Share button with year ===
                movie_title = st.session_state.similar_movies_target

                # Get the year from the first result (the source movie)
                source_movie = results.iloc[0]
                movie_year = None
                if 'year' in source_movie and pd.notna(source_movie['year']):
                    movie_year = int(source_movie['year'])

                # Generate URL with year for disambiguation
                if movie_year:
                    share_url = generate_share_url(
                        'similar', movie=movie_title, year=movie_year
                    )
                else:
                    share_url = generate_share_url('similar', movie=movie_title)

                display_share_button(share_url)
                # === END NEW ===
```

### 6. Update search functions to accept year parameter:

You'll need to modify the search functions in `search.py` to accept optional `year` parameters:

```python
def similar_movies_search(
    movie_title: str,
    model: "SentenceTransformer",
    index: "faiss.IndexFlatIP",
    movies_df: pd.DataFrame,
    k: int = 100,
    year: Optional[int] = None,  # NEW parameter
) -> pd.DataFrame:
    """Find movies similar to a given movie."""
    # Find the movie by title (case-insensitive)
    movie_matches = movies_df[movies_df["title"].str.lower() == movie_title.lower()]

    # NEW: Filter by year if provided
    if year is not None and len(movie_matches) > 0:
        movie_matches = movie_matches[movie_matches["year"] == year]

    # If no matches with year, fall back to original behavior
    if len(movie_matches) == 0:
        movie_matches = movies_df[movies_df["title"].str.lower() == movie_title.lower()]

    # ... rest of existing logic ...
```

Similar updates needed for `blend_movies()` and `contrastive_search()`.

## Next Steps

1. Review and validate this plan
2. Create implementation tasks in todo list
3. **Update search functions** to accept optional year parameters (search.py)
4. Set up basic URL parameter handling (app.py)
5. Implement share button UI
6. Test thoroughly with different search modes
7. Handle edge cases (duplicate titles, special characters, long queries)
8. Deploy and monitor usage
