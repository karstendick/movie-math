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
- **Search type** (semantic search vs similar movies)
- **Query text** (for semantic search)
- **Selected movie** (for similar movies search)
- **Number of results** (k value)
- **Tab state** (which tab is active)

### Optional/Future State:
- Scroll position to specific movie
- Expanded/collapsed state of movie cards
- Specific movie highlighted

---

## Technical Approach

### Option 1: URL Query Parameters
**Pros:**
- Simple to implement
- Human-readable
- Easy to debug
- No backend required

**Cons:**
- URL length limits (~2000 chars)
- Spaces and special characters need encoding

**Implementation:**
```
https://moviemath.com/?mode=semantic&q=movies+about+time+travel&k=10&tab=results
https://moviemath.com/?mode=similar&movie=The+Matrix&movieId=603&k=10
```

### Option 2: Base64 Encoded State
**Pros:**
- Compact URLs
- Can encode more complex state
- Still no backend required

**Cons:**
- Not human-readable
- Slightly more complex encoding/decoding

**Implementation:**
```
https://moviemath.com/?share=eyJtb2RlIjoic2VtYW50aWMiLCJxIjoidGltZSB0cmF2ZWwiLCJrIjoxMH0=
```

### Option 3: Short Links with Backend
**Pros:**
- Very short URLs
- Can track analytics
- Can update/expire links

**Cons:**
- Requires backend/database
- More complex implementation
- Persistence concerns

**Implementation:**
```
https://moviemath.com/s/abc123
```

### **Recommended: Start with Option 1, migrate to Option 3 if needed**

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
  'movie1': str,         # first movie (blend mode)
  'movie2': str,         # second movie (blend mode)
  'like': str,           # movie to match (contrastive mode)
  'avoid': str,          # aspect to avoid (contrastive mode)
}
```

Example URLs:
- Semantic: `?mode=semantic&q=atmospheric+sci-fi+with+stunning+visuals`
- Similar: `?mode=similar&movie=The+Matrix`
- Blend: `?mode=blend&movie1=Inception&movie2=The+Matrix`
- Contrastive: `?mode=contrastive&like=The+Godfather&avoid=violent`

### Phase 2: Enhanced Sharing Features

- [ ] Social media meta tags (OpenGraph, Twitter Cards)
- [ ] Preview image generation for shared links
- [ ] Share to Twitter/Facebook/LinkedIn buttons
- [ ] QR code generation for mobile sharing

### Phase 3: Short Links (Optional)

- [ ] Backend API for creating short links
- [ ] Database schema for link storage
- [ ] Analytics tracking (views, clicks)
- [ ] Link expiration/management

---

## UI/UX Design

### Share Button Placement
- **Option A**: Next to search stats ("Searched 20,000 movies in 0.4 seconds")
- **Option B**: Floating action button in bottom right
- **Option C**: In header/toolbar area

### Share Flow
1. User clicks "Share" button
2. URL is generated from current state
3. URL is copied to clipboard automatically
4. Show success message: "Link copied! Share your results."
5. Optional: Show modal with link preview and social sharing options

### Visual Feedback
- Icon: 🔗 or share icon (⤴)
- Tooltip: "Share these results"
- Success state: Checkmark or "Copied!" message

---

## Edge Cases to Handle

1. **No Search Performed**: Disable/hide share button
2. **Long Query Strings**: URL too long → truncate or switch to base64
3. **Movie Not Found**: Shared similar movie link with invalid movie ID
4. **Browser Compatibility**: Clipboard API not available in some browsers
5. **Security**: Sanitize/validate URL parameters to prevent XSS
6. **Stale Data**: Movie removed from index but shared link references it

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

---

## Success Metrics

- Number of shares generated
- Click-through rate on shared links
- Bounce rate on shared link visits
- User feedback on sharing feature

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
Two options for clipboard functionality:

**Option A: JavaScript with streamlit-clipboard component**
```python
import streamlit.components.v1 as components

def create_share_button(share_url):
    components.html(f"""
        <button onclick="navigator.clipboard.writeText('{share_url}')">
            Share Results
        </button>
        <script>
        function copyToClipboard() {{
            navigator.clipboard.writeText('{share_url}')
                .then(() => alert('Link copied!'));
        }}
        </script>
    """)
```

**Option B: Display URL with manual copy**
```python
col1, col2 = st.columns([4, 1])
with col1:
    st.code(share_url, language=None)
with col2:
    st.button("Copy", help="Copy link to clipboard")
```

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

### Current Stack
- **Streamlit** (native query params support)
- **Python urllib** (URL encoding/decoding)
- **Python standard library** (base64 for optional encoding)

### Optional Libraries
- **streamlit-clipboard** (for better clipboard UX)
- **pyshorteners** (for URL shortening, Phase 3)

### Optional Backend (Phase 3)
- FastAPI or Flask API
- Database (PostgreSQL, MongoDB, etc.)
- Short URL generation
- Analytics service

---

## Open Questions

1. Should we encode the actual search results or just the query? (Just query → re-run search)
2. Should shared links work if the index changes? (Yes, but results may differ)
3. Do we need authentication/user accounts for tracking shares? (No for MVP)
4. Should we add a "report inappropriate content" for shared links? (Future consideration)

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
        if movie:
            st.session_state.active_tab = "🎯 Similar Movies"
            st.session_state.similar_movies_target = movie
            # Auto-trigger search
            results = similar_movies_search(movie, model, index, movies_df, k=100)
            st.session_state.similar_movies_results = results

    # ... handle other modes (blend, contrastive) ...
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

## Next Steps

1. Review and validate this plan
2. Create implementation tasks in todo list
3. Set up basic URL parameter handling
4. Implement share button UI
5. Test thoroughly with different search modes
6. Handle edge cases (special characters, long queries)
7. Deploy and monitor usage
