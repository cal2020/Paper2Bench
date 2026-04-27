"""Search arXiv and download paper PDFs."""

import argparse
import os
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from typing import Optional

try:
    import arxiv
except ImportError:
    sys.exit("Missing dependency: pip install arxiv")


def _safe_print(s: str) -> None:
    """Print a string that may contain chars unencodable on the current console
    (e.g. cp1252 on Windows can't render Greek letters from arxiv titles)."""
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(s.encode(enc, errors="replace").decode(enc))


def _normalize_title(t: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- for fuzzy matching."""
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def _split_query(query: str) -> tuple[str, str]:
    """Split 'Title (Author et al.)' into (title, author_hint).

    The parenthetical author suffix poisons arxiv's relevance ranking
    (the literal token 'et al' floods unrelated papers), so we strip
    it from the search query and use the author last name only as a
    soft re-ranking hint.
    """
    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", query.strip())
    if not m:
        return query.strip(), ""
    title = m.group(1).strip()
    author = re.sub(r"\bet\s+al\.?\b", "", m.group(2), flags=re.I).strip(" ,.;")
    return title, author


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
    max_results: int = 10,
    min_similarity: float = 0.75,
) -> Optional[str]:
    """
    Search arXiv for a paper, optionally confirm with user, and download PDF.
    Returns the output path on success, None on failure.

    In ``auto`` mode, results are re-ranked by title similarity against the
    query title rather than blindly picking arxiv's #1 -- arxiv's relevance
    sort is unreliable for short title-style queries (a single noisy hit can
    bump the real paper out of the top slot).
    """
    title_query, author_hint = _split_query(query)

    # Strategy: title-field search first (high precision for known titles),
    # fall back to free-text if it returns nothing.
    ti_query = f'ti:"{title_query}"'
    if author_hint:
        ti_query += f" AND au:{author_hint.split()[-1]}"
    results = search_arxiv(ti_query, max_results=max_results)
    if not results:
        cleaned_query = f"{title_query} {author_hint}".strip()
        print(f"[search] ti: returned 0 results, falling back to free-text: {cleaned_query!r}")
        results = search_arxiv(cleaned_query, max_results=max_results)

    if not results:
        print(f"No results found for query: {query}")
        return None

    if auto:
        norm_target = _normalize_title(title_query)
        norm_author = author_hint.lower()
        scored = []
        for r in results:
            title_sim = SequenceMatcher(
                None, norm_target, _normalize_title(r["title"])
            ).ratio()
            author_bonus = 0.1 if norm_author and any(
                norm_author in a.lower() for a in r["authors"]
            ) else 0.0
            scored.append((title_sim + author_bonus, title_sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_title_sim, selected = scored[0]
        _safe_print(f"\n[auto] best match: {selected['title'][:90]}")
        _safe_print(f"[auto] title-similarity={best_title_sim:.2f}  combined={best_score:.2f}")
        if best_title_sim < min_similarity:
            _safe_print(f"[auto] ABORT: best match below similarity threshold "
                  f"({best_title_sim:.2f} < {min_similarity}).")
            _safe_print(f"[auto] top candidates:")
            for sc, ts, r in scored[:5]:
                _safe_print(f"   sim={ts:.2f}  {r['title'][:90]}")
            return None
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
        _safe_print(f"\nSelected: {selected['title']}")

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
