# crawler/storage.py
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Iterable


# define standard data structure, then add data persistence 
# NOTE: this is a QUEUE and is not supposed to store/be used for queries 
@dataclass
class FrontierItem:
    url: str
    discovered_from: str
    depth: int = 0
    priority: int = 0  # we can add scoring later

def write_frontier_jsonl(path: str, items: Iterable[FrontierItem]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
