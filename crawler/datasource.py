from __future__ import annotations
import yaml
from typing import List
from crawler.filters import canonicalize_url, is_allowed_domain, looks_like_doc_url
from crawler.sitemap import candidate_sitemap_urls, expand_sitemaps
from crawler.storage import FrontierItem, write_frontier_jsonl

def load_seeds(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["seeds"]

def build_frontier(seeds_path: str, out_path: str) -> None:
    seeds = load_seeds(seeds_path)
    allowed_domains = {s["domain"].lower() for s in seeds}

    frontier: List[FrontierItem] = []
    seen_urls = set()

    for seed in seeds:
        for start in seed["start_urls"]:
            # 1) discover likely sitemaps
            sm_candidates = candidate_sitemap_urls(start)

            # 2) expand sitemap(s) into page URLs
            pages = expand_sitemaps(sm_candidates)

            # 3) filter pages into docs-like URLs
            for url in pages:
                url = canonicalize_url(url)
                if url in seen_urls:
                    continue
                if not is_allowed_domain(url, allowed_domains):
                    continue
                if not looks_like_doc_url(url):
                    continue

                seen_urls.add(url)
                frontier.append(FrontierItem(url=url, discovered_from=start, depth=0, priority=0))

    write_frontier_jsonl(out_path, frontier)
    print(f"Wrote {len(frontier)} frontier URLs to {out_path}")

if __name__ == "__main__":
    build_frontier("crawler/seeds.yaml", "crawler/frontier.jsonl")
