"""
Multithreaded crawler — processes frontier.jsonl with concurrent workers.

Usage:
    python main_threaded.py --workers 8 --batch-size 50
    python main_threaded.py --frontier frontier-queue/frontier.jsonl --limit 1000 --workers 16
"""

from __future__ import annotations
import argparse
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import List, Set
from dataclasses import dataclass

from pipeline.fetcher.fetcher import Fetcher
from pipeline.fetcher.normalizer import normalize_html
from pipeline.chunker.chunker import chunk
from pipeline.embedder.embedder import embed_and_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VISITED_PATH = Path("visited.txt")
DEFAULT_FRONTIER = "frontier-queue/frontier.jsonl"
DELAY_SECONDS = 0.1  # reduced delay for multithreaded processing

@dataclass
class CrawlItem:
    url: str
    item_data: dict

class ThreadSafeVisitedTracker:
    """Thread-safe tracker for visited URLs."""
    
    def __init__(self, visited_path: Path):
        self.visited_path = visited_path
        self._visited: Set[str] = self._load_visited()
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
    
    def _load_visited(self) -> Set[str]:
        if self.visited_path.exists():
            return set(self.visited_path.read_text().splitlines())
        return set()
    
    def is_visited(self, url: str) -> bool:
        with self._lock:
            return url in self._visited
    
    def mark_visited(self, url: str) -> None:
        with self._lock:
            if url not in self._visited:
                self._visited.add(url)
                # Write to file immediately for persistence
                with self._file_lock:
                    with self.visited_path.open("a") as f:
                        f.write(url + "\n")

def load_frontier_items(frontier_path: str, visited_tracker: ThreadSafeVisitedTracker, limit: int | None = None) -> List[CrawlItem]:
    """Load all unvisited items from frontier file."""
    items = []
    with open(frontier_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if limit and len(items) >= limit:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                url = item.get("url", "").strip()
                
                if not url:
                    continue
                    
                if not visited_tracker.is_visited(url):
                    items.append(CrawlItem(url=url, item_data=item))
                    
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed frontier line %d: %s", line_num, e)
    
    logger.info("Loaded %d unvisited items from frontier", len(items))
    return items

def process_single_url(crawl_item: CrawlItem, visited_tracker: ThreadSafeVisitedTracker) -> tuple[bool, str]:
    """Process a single URL. Returns (success, error_message)."""
    url = crawl_item.url
    fetcher = Fetcher()
    
    try:
        # Check again in case another thread processed it
        if visited_tracker.is_visited(url):
            return True, "already_visited"
            
        logger.debug("Fetching %s", url)
        r = fetcher.fetch(url)
        normalized = normalize_html(r.html, r.url)

        if not normalized.blocks:
            logger.debug("No blocks extracted from %s", url)
            visited_tracker.mark_visited(url)
            return True, "no_blocks"

        chunks = chunk(normalized)

        if not chunks:
            logger.debug("No chunks produced from %s", url)
            visited_tracker.mark_visited(url)
            return True, "no_chunks"

        embed_and_store(chunks)
        visited_tracker.mark_visited(url)
        
        logger.info("✓ Processed %s → %d chunks", url, len(chunks))
        time.sleep(DELAY_SECONDS)  # Be respectful to servers
        
        return True, f"success_{len(chunks)}_chunks"

    except Exception as e:
        logger.error("✗ Error processing %s: %s", url, e)
        visited_tracker.mark_visited(url)  # Mark as visited to avoid infinite retries
        return False, str(e)

def process_batch(batch: List[CrawlItem], visited_tracker: ThreadSafeVisitedTracker) -> dict:
    """Process a batch of URLs and return statistics."""
    stats = {"processed": 0, "errors": 0, "skipped": 0}
    
    for crawl_item in batch:
        success, message = process_single_url(crawl_item, visited_tracker)
        
        if success:
            if message == "already_visited":
                stats["skipped"] += 1
            else:
                stats["processed"] += 1
        else:
            stats["errors"] += 1
    
    return stats

def crawl_multithreaded(
    frontier_path: str, 
    limit: int | None = None,
    workers: int = 8,
    batch_size: int = 20
) -> None:
    """Main multithreaded crawling function."""
    
    logger.info("Starting multithreaded crawler with %d workers, batch size %d", workers, batch_size)
    
    visited_tracker = ThreadSafeVisitedTracker(VISITED_PATH)
    crawl_items = load_frontier_items(frontier_path, visited_tracker, limit)
    
    if not crawl_items:
        logger.info("No unvisited items to process")
        return
    
    # Create batches
    batches = []
    for i in range(0, len(crawl_items), batch_size):
        batch = crawl_items[i:i + batch_size]
        batches.append(batch)
    
    logger.info("Created %d batches from %d items", len(batches), len(crawl_items))
    
    # Process batches with thread pool
    total_stats = {"processed": 0, "errors": 0, "skipped": 0}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_batch, batch, visited_tracker): batch 
            for batch in batches
        }
        
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            completed_batches += 1
            
            try:
                batch_stats = future.result()
                
                # Update totals
                for key in total_stats:
                    total_stats[key] += batch_stats[key]
                
                # Log progress
                elapsed = time.time() - start_time
                rate = total_stats["processed"] / elapsed if elapsed > 0 else 0
                
                logger.info(
                    "Batch %d/%d complete | Processed: %d, Errors: %d, Skipped: %d | Rate: %.1f/sec",
                    completed_batches, len(batches),
                    total_stats["processed"], total_stats["errors"], total_stats["skipped"],
                    rate
                )
                
            except Exception as e:
                logger.error("Batch failed: %s", e)
                total_stats["errors"] += len(batch)
    
    elapsed = time.time() - start_time
    final_rate = total_stats["processed"] / elapsed if elapsed > 0 else 0
    
    logger.info(
        "Crawling complete! Processed: %d, Errors: %d, Skipped: %d | "
        "Total time: %.1fs, Average rate: %.1f URLs/sec",
        total_stats["processed"], total_stats["errors"], total_stats["skipped"],
        elapsed, final_rate
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multithreaded web crawler")
    parser.add_argument("--frontier", default=DEFAULT_FRONTIER, help="Path to frontier.jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="Max URLs to process")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--batch-size", type=int, default=20, help="URLs per batch")
    args = parser.parse_args()

    crawl_multithreaded(args.frontier, args.limit, args.workers, args.batch_size)