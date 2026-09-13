"""arXiv paper fetching (Phase 2).

Downloads real academic paper metadata from the free arXiv API.
No API key required.
"""
from __future__ import annotations

import time

import arxiv

# Be polite to arXiv: their API asks for >= 3s between requests from
# the same client (handled internally by arxiv.Client, but we set it
# explicitly so it survives library defaults changing).
ARXIV_CLIENT = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)


def fetch_arxiv_papers(query: str, max_results: int = 25) -> list[dict]:
    """Fetch papers from arXiv matching a query string.

    Returns a list of dicts with keys: title, abstract, authors,
    published, url, categories.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: list[dict] = []
    for result in ARXIV_CLIENT.results(search):
        papers.append(
            {
                "title": " ".join(result.title.split()),  # collapse whitespace
                "abstract": " ".join(result.summary.split()),
                "authors": [a.name for a in result.authors],
                "published": str(result.published),
                "url": result.entry_id,
                "categories": list(result.categories),
            }
        )
    return papers


if __name__ == "__main__":
    q = input("arXiv search query: ") or "retrieval augmented generation"
    found = fetch_arxiv_papers(q, max_results=5)
    if not found:
        print("No papers found — check your internet connection.")
    for i, p in enumerate(found, 1):
        print(f"{i}. {p['title']}\n   {p['url']}")
