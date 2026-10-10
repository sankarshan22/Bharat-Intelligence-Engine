from fastmcp import FastMCP
import requests
try:
    from config import gnews_api
except ImportError:
    from mcp_servers.geopolitics.config import gnews_api


# ── SERVER SETUP ───────────────────────────────────────────────────────────────

news_mcp = FastMCP(name="News Server")
GNEWS_BASE = "https://gnews.io/api/v4"

API_KEY=gnews_api

VALID_CATEGORIES = [
    "general", "world", "nation", "business",
    "technology", "entertainment", "sports", "science", "health"
]


# ── SHARED HELPER ──────────────────────────────────────────────────────────────

def _fetch_news(endpoint: str, params: dict) -> dict:
    """Internal helper to call GNews API and handle errors."""
    params["apikey"] = API_KEY
    try:
        response = requests.get(f"{GNEWS_BASE}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e.response.status_code} - {e.response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


# ── TOOL 1: get_top_headlines ──────────────────────────────────────────────────

@news_mcp.tool
async def get_top_headlines(
    category: str = "general",
    lang: str = "en",
    country: str = "us",
    max_results: int = 10
) -> dict:
    """
    Fetch top headlines from GNews by category.

    Args:
        category: News category. One of: general, world, nation, business,
                  technology, entertainment, sports, science, health.
                  Defaults to 'general'.
        lang:     Language code (e.g. 'en', 'fr', 'de'). Defaults to 'en'.
        country:  Country code (e.g. 'us', 'in', 'gb'). Defaults to 'us'.
        max_results: Number of articles to return (1–10). Defaults to 10.

    Returns:
        A dict with 'totalArticles' and a list of 'articles', each containing
        title, description, url, publishedAt, and source name.
    """
    if category not in VALID_CATEGORIES:
        return {
            "error": f"Invalid category '{category}'. "
                     f"Choose from: {', '.join(VALID_CATEGORIES)}"
        }

    max_results = max(1, min(max_results, 10))  # clamp between 1 and 10

    data = _fetch_news("top-headlines", {
        "category": category,
        "lang": lang,
        "country": country,
        "max": max_results,
    })

    if "error" in data:
        return data

    articles = [
        {
            "title": article.get("title"),
            "description": article.get("description"),
            "url": article.get("url"),
            "publishedAt": article.get("publishedAt"),
            "source": (article.get("source") or {}).get("name"),
        }
        for article in (data.get("articles") or [])
    ]

    return {
        "totalArticles": data.get("totalArticles", 0),
        "category": category,
        "articles": articles,
    }


# ── TOOL 2: search_news ────────────────────────────────────────────────────────

@news_mcp.tool
async def search_news(
    query: str,
    lang: str = "en",
    max_results: int = 10
) -> dict:
    """
    Search for news articles matching a keyword or phrase.

    Args:
        query:       Search keywords (e.g. 'AI', 'cricket world cup').
        lang:        Language code (e.g. 'en', 'hi'). Defaults to 'en'.
        max_results: Number of articles to return (1–10). Defaults to 10.

    Returns:
        A dict with 'totalArticles' and a list of 'articles', each containing
        title, description, url, publishedAt, and source name.
    """
    if not query.strip():
        return {"error": "Search query cannot be empty."}

    max_results = max(1, min(max_results, 10))

    data = _fetch_news("search", {
        "q": query,
        "lang": lang,
        "max": max_results,
    })

    if "error" in data:
        return data

    articles = [
        {
            "title": article.get("title"),
            "description": article.get("description"),
            "url": article.get("url"),
            "publishedAt": article.get("publishedAt"),
            "source": (article.get("source") or {}).get("name"),
        }
        for article in (data.get("articles") or [])
    ]

    return {
        "totalArticles": data.get("totalArticles", 0),
        "query": query,
        "articles": articles,
    }


# ── TOOL 3: get_available_categories ──────────────────────────────────────────

@news_mcp.tool
async def get_available_categories() -> dict:
    """
    List all available news categories supported by the GNews API.

    Returns:
        A dict with a 'categories' key containing a list of valid category names.
    """
    return {"categories": VALID_CATEGORIES}


# ── ENTRYPOINT ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    news_mcp.run(transport="sse", port=8003)