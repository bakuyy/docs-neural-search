"""
Async crawler — maximum performance with asyncio for I/O-bound operations.

Usage:
    python main_async.py --concurrent 50 --batch-size 100
    python main_async.py --frontier frontier-queue/frontier.jsonl --limit 5000 --concurrent 100
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Set
from dataclasses import dataclass
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor

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
DEFAULT_FRONTIER = "pipeline/frontier-queue/frontier.jsonl"

@dataclass
class CrawlItem:
    url: str
    item_data: dict

class AsyncVisitedTracker:
    """Async-safe visited URL tracker."""
    
    def __init__(self, visited_path: Path):
        self.visited_path = visited_path
        self._visited: Set[str] = set()
        self._lock = asyncio.Lock()
        self._loaded = False
    
    async def _load_visited(self):
        if self._loaded:
            return
            
        if self.visited_path.exists():
            async with aiofiles.open(self.visited_path, "r") as f:
                content = await f.read()
                self._visited = set(line.strip() for line in content.splitlines() if line.strip())
        
        self._loaded = True
        logger.info("Loaded %d visited URLs", len(self._visited))
    
    async def is_visited(self, url: str) -> bool:
        await self._load_visited()
        async with self._lock:
            return url in self._visited
    
    async def mark_visited(self, url: str) -> None:
        await self._load_visited()
        async with self._lock:
            if url not in self._visited:
                self._visited.add(url)
                # Append to file
                async with aiofiles.open(self.visited_path, "a") as f:
                    await f.write(f"{url}\n")

async def load_frontier_items(frontier_path: str, visited_tracker: AsyncVisitedTracker, limit: int | None = None) -> List[CrawlItem]:
    """Load unvisited items from frontier file."""
    items = []
    
    async with aiofiles.open(frontier_path, "r") as f:
        async for line_num, line in async_enumerate(f, 1):
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
                    
                if not await visited_tracker.is_visited(url):
                    items.append(CrawlItem(url=url, item_data=item))
                    
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed frontier line %d: %s", line_num, e)
    
    logger.info("Loaded %d unvisited items from frontier", len(items))
    return items

async def async_enumerate(aiter, start=0):
    """Async version of enumerate."""
    n = start
    async for elem in aiter:
        yield n, elem
        n += 1

async def fetch_url_content(session: aiohttp.ClientSession, url: str) -> tuple[bool, str, str]:
    """Fetch URL content. Returns (success, html_content, error_message)."""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                html = await response.text()
                return True, html, ""
            else:
                return False, "", f"HTTP {response.status}"
    except Exception as e:
        return False, "", str(e)

async def process_single_url(
    session: aiohttp.ClientSession, 
    crawl_item: CrawlItem, 
    visited_tracker: AsyncVisitedTracker,
    executor: ThreadPoolExecutor
) -> tuple[bool, str]:
    """Process a single URL asynchronously."""
    url = crawl_item.url
    
    try:
        # Check if already visited
        if await visited_tracker.is_visited(url):
            return True, "already_visited"
        
        # Fetch content asynchronously
        success, html, error_msg = await fetch_url_content(session, url)
        
        if not success:
            logger.debug("Failed to fetch %s: %s", url, error_msg)
            await visited_tracker.mark_visited(url)
            return False, error_msg
        
        # Run CPU-bound operations in thread pool
        loop = asyncio.get_event_loop()
        
        # Normalize HTML
        normalized = await loop.run_in_executor(
            executor, 
            lambda: normalize_html(html, url)
        )
        
        if not normalized.blocks:
            logger.debug("No blocks extracted from %s", url)
            await visited_tracker.mark_visited(url)
            return True, "no_blocks"
        
        # Chunk content
        chunks = await loop.run_in_executor(
            executor,
            lambda: chunk(normalized)
        )
        
        if not chunks:
            logger.debug("No chunks produced from %s", url)
            await visited_tracker.mark_visited(url)
            return True, "no_chunks"
        
        # Embed and store (this involves API calls and DB operations)
        await loop.run_in_executor(
            executor,
            lambda: embed_and_store(chunks)
        )
        
        await visited_tracker.mark_visited(url)
        logger.info("✓ Processed %s → %d chunks", url, len(chunks))
        
        return True, f"success_{len(chunks)}_chunks"
        
    except Exception as e:
        logger.error("✗ Error processing %s: %s", url, e)
        await visited_tracker.mark_visited(url)
        return False, str(e)

async def process_batch_async(
    session: aiohttp.ClientSession,
    batch: List[CrawlItem], 
    visited_tracker: AsyncVisitedTracker,
    executor: ThreadPoolExecutor,
    semaphore: asyncio.Semaphore
) -> dict:
    """Process a batch of URLs concurrently."""
    
    async def process_with_semaphore(item):
        async with semaphore:
            return await process_single_url(session, item, visited_tracker, executor)
    
    # Process all items in batch concurrently
    results = await asyncio.gather(
        *[process_with_semaphore(item) for item in batch],
        return_exceptions=True
    )
    
    stats = {"processed": 0, "errors": 0, "skipped": 0}
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Exception processing %s: %s", batch[i].url, result)
            stats["errors"] += 1
        else:
            success, message = result
            if success:
                if message == "already_visited":
                    stats["skipped"] += 1
                else:
                    stats["processed"] += 1
            else:
                stats["errors"] += 1
    
    return stats

async def crawl_async(
    frontier_path: str,
    limit: int | None = None,
    concurrent: int = 50,
    batch_size: int = 100
) -> None:
    """Main async crawling function."""
    
    logger.info("Starting async crawler with %d concurrent connections, batch size %d", concurrent, batch_size)
    
    visited_tracker = AsyncVisitedTracker(VISITED_PATH)
    crawl_items = await load_frontier_items(frontier_path, visited_tracker, limit)
    
    if not crawl_items:
        logger.info("No unvisited items to process")
        return
    
    # Create batches
    batches = []
    for i in range(0, len(crawl_items), batch_size):
        batch = crawl_items[i:i + batch_size]
        batches.append(batch)
    
    logger.info("Created %d batches from %d items", len(batches), len(crawl_items))
    
    # Setup async components
    connector = aiohttp.TCPConnector(limit=concurrent, limit_per_host=20)
    timeout = aiohttp.ClientTimeout(total=30)
    semaphore = asyncio.Semaphore(concurrent)
    
    # Thread pool for CPU-bound operations
    executor = ThreadPoolExecutor(max_workers=min(concurrent // 4, 20))
    
    total_stats = {"processed": 0, "errors": 0, "skipped": 0}
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Process batches
            for batch_num, batch in enumerate(batches, 1):
                batch_stats = await process_batch_async(session, batch, visited_tracker, executor, semaphore)
                
                # Update totals
                for key in total_stats:
                    total_stats[key] += batch_stats[key]
                
                # Log progress
                elapsed = time.time() - start_time
                rate = total_stats["processed"] / elapsed if elapsed > 0 else 0
                
                logger.info(
                    "Batch %d/%d complete | Processed: %d, Errors: %d, Skipped: %d | Rate: %.1f/sec",
                    batch_num, len(batches),
                    total_stats["processed"], total_stats["errors"], total_stats["skipped"],
                    rate
                )
    
    finally:
        executor.shutdown(wait=True)
    
    elapsed = time.time() - start_time
    final_rate = total_stats["processed"] / elapsed if elapsed > 0 else 0
    
    logger.info(
        "Crawling complete! Processed: %d, Errors: %d, Skipped: %d | "
        "Total time: %.1fs, Average rate: %.1f URLs/sec",
        total_stats["processed"], total_stats["errors"], total_stats["skipped"],
        elapsed, final_rate
    )

def main():
    parser = argparse.ArgumentParser(description="Async web crawler for maximum performance")
    parser.add_argument("--frontier", default=DEFAULT_FRONTIER, help="Path to frontier.jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="Max URLs to process")
    parser.add_argument("--concurrent", type=int, default=50, help="Max concurrent connections")
    parser.add_argument("--batch-size", type=int, default=100, help="URLs per batch")
    args = parser.parse_args()
    
    # Run the async crawler
    asyncio.run(crawl_async(args.frontier, args.limit, args.concurrent, args.batch_size))

if __name__ == "__main__":
    main()