from __future__ import annotations
import os
import sys
import logging
import yaml
from typing import List

# Allow running this file directly from the crawler dir.
if __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

from crawler.filters import canonicalize_url, is_allowed_domain, is_valid_doc_url
from crawler.sitemap import candidate_sitemap_urls, expand_sitemaps
from crawler.storage import FrontierItem, write_frontier_jsonl

logger = logging.getLogger(__name__)

def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def load_seeds(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["seeds"]

def build_frontier(seeds_path: str, out_path: str) -> None:
    _configure_logging()
    seeds = load_seeds(seeds_path)
    allowed_domains = {s["domain"].lower() for s in seeds}

    frontier: List[FrontierItem] = []
    seen_urls = set()

    for seed in seeds:
        domain = seed.get("domain", "")
        start_urls = seed.get("start_urls", [])
        logger.info("Seed %s (%d start urls)", domain, len(start_urls))
        for start in seed["start_urls"]:
            logger.info("Start url: %s", start)
            # 1) discover likely sitemaps
            sm_candidates = candidate_sitemap_urls(start)
            logger.info("Sitemap candidates: %s", ", ".join(sm_candidates))

            # 2) expand sitemap(s) into page URLs
            pages = expand_sitemaps(sm_candidates)
            logger.info("Discovered %d pages from sitemaps", len(pages))

            # 3) filter pages into docs-like URLs
            kept = 0
            for url in pages:
                url = canonicalize_url(url)
                if url in seen_urls:
                    continue
                if not is_allowed_domain(url, allowed_domains):
                    continue
                if not is_valid_doc_url(url):
                    continue

                seen_urls.add(url)
                frontier.append(FrontierItem(url=url, discovered_from=start, depth=0, priority=0))
                kept += 1
            logger.info("Kept %d urls for seed %s", kept, domain)

    write_frontier_jsonl(out_path, frontier)
    print(f"Wrote {len(frontier)} frontier URLs to {out_path}")

if __name__ == "__main__":
    build_frontier("crawler/seeds.yaml", "crawler/frontier.jsonl")
