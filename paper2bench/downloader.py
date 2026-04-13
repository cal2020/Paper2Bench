"""Search arXiv and download paper PDFs."""

import argparse
import os
import sys
import time
import urllib.request
from typing import Optional

try:
    import arxiv
except ImportError:
    sys.exit("Missing dependency: pip install arxiv")


# Backoff schedule for arxiv HTTP 429 rate-limits.
_BACKOFF_SECONDS = (5, 20, 60)


def search_arxiv(query: str, max_results: int = 5) -> list:
    """Search arXiv by title/author and return a list of result dicts.

    Retries with exponential backoff on HTTP 429 (rate limit) errors from
    the arxiv API, which are common when multiple jobs query in parallel.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    last_err = None
    for attempt, delay in enumerate((0,) + _BACKOFF_SECONDS):
        if delay:
            print(f"[arxiv] rate-limited, retrying in {delay}s "
                  f"(attempt {attempt}/{len(_BACKOFF_SECONDS)}) ...")
            time.sleep(delay)
        try:
            results = []
            for r in client.results(search):
                results.append({
                    "title": r.title,
                    "authors": [a.name for a in r.authors],
                    "published": r.published.strftime("%Y-%m-%d"),
                    "arxiv_id": r.entry_id.split("/")[-1],
                    "pdf_url": r.pdf_url,
                    "abstract": r.summary[:300],
                })
            return results
        except arxiv.HTTPError as e:
            last_err = e
            if getattr(e, "status", None) != 429:
                raise
            continue

    raise RuntimeError(
        f"arxiv search failed after {len(_BACKOFF_SECONDS)} retries: {last_err}"
    )


def display_results(results: list) -> None:
    """Pretty-print search results to stdout."""
    for i, r in enumerate(results, 1):
        authors = ", ".join(r["authors"][:3])
        if len(r["authors"]) > 3:
            authors += " et al."
        print(f"\n[{i}] {r['title']}")
        print(f"    Authors:   {authors}")
        print(f"    Published: {r['published']}")
        print(f"    arXiv ID:  {r['arxiv_id']}")
        print(f"    Abstract:  {r['abstract']}...")


def download_pdf(pdf_url: str, output_path: str) -> None:
    """Download a PDF from a URL."""
    print(f"Downloading PDF to {output_path} ...")
    urllib.request.urlretrieve(pdf_url, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Downloaded: {size_mb:.1f} MB")


def find_and_download(
    query: str,
    task_id: str,
    output_dir: str = "./papers",
    auto: bool = False,
    max_results: int = 5,
) -> Optional[str]:
    """
    Search arXiv for a paper, optionally confirm with user, and download PDF.
    Returns the output path on success, None on failure.
    """
    results = search_arxiv(query, max_results=max_results)

    if not results:
        print(f"No results found for query: {query}")
        return None

    if auto:
        choice = 0
    else:
        display_results(results)
        print(f"\nSelect a paper [1-{len(results)}] or 0 to skip: ", end="")
        try:
            choice = int(input().strip()) - 1
        except (ValueError, EOFError):
            print("Invalid input, skipping.")
            return None
        if choice < 0 or choice >= len(results):
            print("Skipped.")
            return None

    selected = results[choice]
    print(f"\nSelected: {selected['title']}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{task_id}.pdf")
    download_pdf(selected["pdf_url"], output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Search arXiv and download a paper PDF."
    )
    parser.add_argument("query", help="Search query (paper title and/or author names)")
    parser.add_argument("--task-id", required=True, help="Task ID for naming the output PDF")
    parser.add_argument("-o", "--output-dir", default="./papers", help="Output directory (default: ./papers)")
    parser.add_argument("--auto", action="store_true", help="Auto-select first result")
    parser.add_argument("--max-results", type=int, default=5, help="Max search results (default: 5)")
    args = parser.parse_args()

    result = find_and_download(
        query=args.query,
        task_id=args.task_id,
        output_dir=args.output_dir,
        auto=args.auto,
        max_results=args.max_results,
    )
    if result:
        print(f"\nPDF saved to: {result}")
    else:
        print("\nNo PDF downloaded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
