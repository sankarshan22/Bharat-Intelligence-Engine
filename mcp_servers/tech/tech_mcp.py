from fastmcp import FastMCP
import requests
try:
    from config import guardian_tech_api
except ImportError:
    from mcp_servers.tech.config import guardian_tech_api
from typing import Optional
from datetime import datetime, timedelta

tech_mcp = FastMCP(name="Tech Server")

BASE_URL = "https://content.guardianapis.com"


@tech_mcp.tool
async def search_tech_news(
    query: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page_size: int = 5
) -> dict:
    """
    Search for tech news articles using the Guardian API.
    Use this for broad topic searches. Run multiple times with different queries for comprehensive coverage.
    Args:
        query: Search term or topic (e.g. 'artificial intelligence', 'India semiconductor policy')
        from_date: Start date in YYYY-MM-DD format. Defaults to 30 days ago if not provided.
        to_date: End date in YYYY-MM-DD format (optional)
        page_size: Number of results to return (default 5, use 8 for broad queries, max 50)
    """
    # Default from_date to 30 days ago so results are always recent
    if not from_date:
        from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "section": "technology",
        "api-key": guardian_tech_api,
        "page-size": page_size,
        "show-fields": "headline,trailText,bodyText,shortUrl,byline,firstPublicationDate",
        "order-by": "newest",
        "from-date": from_date,
    }

    if to_date:
        params["to-date"] = to_date

    response = requests.get(f"{BASE_URL}/search", params=params)
    data = response.json()

    try:
        results = data["response"]["results"]
        if not results:
            return {"message": f"No tech news found for query: {query}"}

        articles = []
        for article in results:
            fields = article.get("fields") or {}
            # trailText is a clean 1-2 sentence summary — prefer it over slicing bodyText
            summary = fields.get("trailText") or fields.get("bodyText", "")[:300]
            articles.append({
                "id": article.get("id", ""),
                "headline": fields.get("headline", article.get("webTitle", "")),
                "published": fields.get("firstPublicationDate", ""),
                "author": fields.get("byline", ""),
                "summary": summary,
                "url": fields.get("shortUrl", article.get("webUrl", ""))
            })

        return {"articles": articles}

    except (KeyError, TypeError):
        return {"error": "Unexpected response format", "raw": str(data)[:500]}


@tech_mcp.tool
async def get_article(article_id: str) -> dict:
    """
    Fetch the full body text of a single Guardian article by its ID.
    Use this when search_tech_news returns an interesting headline but you need the full story.
    The article_id comes from the 'id' field in search_tech_news results.
    Args:
        article_id: Guardian article path ID (e.g. 'technology/2026/mar/01/openai-funding-round')
    """
    params = {
        "api-key": guardian_tech_api,
        "show-fields": "headline,trailText,bodyText,byline,firstPublicationDate,shortUrl",
    }

    response = requests.get(f"{BASE_URL}/{article_id}", params=params)
    data = response.json()

    try:
        content = data["response"]["content"]
        fields = content.get("fields") or {}
        return {
            "headline": fields.get("headline", content.get("webTitle", "")),
            "published": fields.get("firstPublicationDate", ""),
            "author": fields.get("byline", ""),
            "body": fields.get("bodyText", ""),
            "url": fields.get("shortUrl", content.get("webUrl", ""))
        }
    except (KeyError, TypeError):
        return {"error": "Article not found or unexpected response", "raw": str(data)[:500]}




@tech_mcp.tool
async def search_by_tag(
    tag: str,
    page_size: int = 5,
    from_date: Optional[str] = None,
) -> dict:
    """
    Search Guardian articles by an exact taxonomy tag for precise topic targeting.
    Use this when you need articles strictly about a specific entity or topic, not just keyword matches.
    More precise than search_tech_news for well-known entities.
    Common useful tags:
      - technology/artificial-intelligence
      - technology/openai
      - technology/google
      - technology/apple
      - technology/india
      - technology/semiconductors
      - technology/cyber-security
      - technology/robotics
    Args:
        tag: Guardian tag ID (e.g. 'technology/artificial-intelligence', 'technology/openai')
        page_size: Number of results to return (default 5, max 50)
        from_date: Start date in YYYY-MM-DD format. Defaults to 30 days ago if not provided.
    """
    if not from_date:
        from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        "tag": tag,
        "api-key": guardian_tech_api,
        "page-size": page_size,
        "show-fields": "headline,trailText,bodyText,shortUrl,byline,firstPublicationDate",
        "order-by": "newest",
        "from-date": from_date,
    }

    response = requests.get(f"{BASE_URL}/search", params=params)
    data = response.json()

    try:
        results = data["response"]["results"]
        if not results:
            return {"message": f"No articles found for tag: {tag}"}

        articles = []
        for article in results:
            fields = article.get("fields") or {}
            summary = fields.get("trailText") or fields.get("bodyText", "")[:300]
            articles.append({
                "id": article.get("id", ""),
                "headline": fields.get("headline", article.get("webTitle", "")),
                "published": fields.get("firstPublicationDate", ""),
                "author": fields.get("byline", ""),
                "summary": summary,
                "url": fields.get("shortUrl", article.get("webUrl", ""))
            })

        return {"articles": articles}

    except (KeyError, TypeError):
        return {"error": "Unexpected response format", "raw": str(data)[:500]}
    


    
if __name__ == "__main__":
    tech_mcp.run(transport="sse", port=8002)