"""
Crawl loop — reads frontier.jsonl and populates the vector DB.

Usage:
    python crawl.py
    python crawl.py --frontier frontier-queue/frontier.jsonl --limit 500
"""

from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

from pipeline.fetcher.fetcher import Fetcher
from pipeline.fetcher.normalizer import normalize_html
from pipeline.chunker.chunker import chunk
from pipeline.embedder.embedder import embed_and_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VISITED_PATH = Path("visited.txt")
DEFAULT_FRONTIER = "frontier-queue/frontier.jsonl"
DELAY_SECONDS = 0.5   # be polite, don't hammer servers


def load_visited() -> set[str]:
    if VISITED_PATH.exists():
        return set(VISITED_PATH.read_text().splitlines())
    return set()


def mark_visited(url: str) -> None:
    with VISITED_PATH.open("a") as f:
        f.write(url + "\n")


def iter_frontier(path: str):
    """Yield FrontierItems from a .jsonl file one at a time (memory efficient)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed frontier line: %s", e)


def crawl(frontier_path: str, limit: int | None = None) -> None:
    visited = load_visited()
    fetcher = Fetcher()

    processed = 0
    skipped = 0
    errors = 0

    for item in iter_frontier(frontier_path):
        if limit and processed >= limit:
            logger.info("Reached limit of %d URLs", limit)
            break

        url = item.get("url", "").strip()
        if not url:
            continue
        if url in visited:
            skipped += 1
            continue

        logger.info("[%d] Fetching %s", processed + 1, url)

        try:
            r = fetcher.fetch(url)
            normalized = normalize_html(r.html, r.url)

            if not normalized.blocks:
                logger.warning("No blocks extracted from %s, skipping", url)
                mark_visited(url)
                visited.add(url)
                continue

            chunks = chunk(normalized)

            if not chunks:
                logger.warning("No chunks produced from %s, skipping", url)
                mark_visited(url)
                visited.add(url)
                continue

            embed_and_store(chunks)
            mark_visited(url)
            visited.add(url)
            processed += 1

            logger.info("  → %d chunks stored", len(chunks))

        except Exception as e:
            logger.error("Error processing %s: %s", url, e)
            errors += 1
            # Still mark visited so we don't retry endlessly in the same run
            mark_visited(url)
            visited.add(url)

        time.sleep(DELAY_SECONDS)

    logger.info("Done. processed=%d skipped=%d errors=%d", processed, skipped, errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", default=DEFAULT_FRONTIER)
    parser.add_argument("--limit", type=int, default=None, help="Max URLs to process")
    args = parser.parse_args()

    crawl(args.frontier, args.limit)