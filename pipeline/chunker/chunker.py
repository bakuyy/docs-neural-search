'''
chunker after we normalized

strategy:
- heading blocks determine new chunks, the heading level determines the hierarchy
- code blocks should be attached to the previous paragraph and should not be split individually
- chunks that exceed the budget are split at the paragraph boundaries (sentence splits as a last resort)
- overlap?
    - given the small semantic units that are already anchored to the headings, it doesn't make sense to overlap them

'''


from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

from pipeline.fetcher.normalizer import Block, NormalizedDoc


MAX_TOKENS = 512
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Chunk:
    heading: str                # nearest ancestor heading text
    heading_level: int          # 1-4
    heading_path: List[str]     # breadcrumb, e.g. ["7.2", "7.2.1", "7.2.1.1"]
    blocks: List[Block]
    canonical_url: str
    title: str

    @property
    def text(self) -> str:
        """Flat text representation used for embedding."""
        parts = []
        if self.heading_path:
            parts.append(" > ".join(self.heading_path))
        for b in self.blocks:
            parts.append(b.text)
        return "\n\n".join(parts)

    @property
    def token_estimate(self) -> int:
        return _estimate_tokens(self.text)


def _split_sentences(text: str) -> List[str]:
    """Naive sentence splitter — good enough for technical prose."""
    text = re.sub(r'\b(e\.g|i\.e|etc|vs|fig|sec|no|vol|pp|Dr|Mr|Mrs|Prof)\.\s',
                  lambda m: m.group().replace('. ', '.<PROTECTED>'), text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.replace('<PROTECTED>', ' ') for s in sentences if s.strip()]


def _split_block_by_sentences(block: Block, budget: int) -> List[Block]:
    """Break a paragraph block into smaller blocks at sentence boundaries."""
    sentences = _split_sentences(block.text)
    sub_blocks = []
    current_sentences: List[str] = []
    current_chars = 0

    for sent in sentences:
        sent_chars = len(sent)
        if current_sentences and (current_chars + sent_chars) // CHARS_PER_TOKEN > budget:
            sub_blocks.append(Block(type=block.type,
                                    text=" ".join(current_sentences),
                                    meta=block.meta))
            current_sentences = [sent]
            current_chars = sent_chars
        else:
            current_sentences.append(sent)
            current_chars += sent_chars + 1

    if current_sentences:
        sub_blocks.append(Block(type=block.type,
                                text=" ".join(current_sentences),
                                meta=block.meta))
    return sub_blocks


def _flush(
    accumulated: List[Block],
    heading: str,
    heading_level: int,
    heading_path: List[str],
    url: str,
    title: str,
    max_tokens: int,
) -> List[Chunk]:
    """
    Turn accumulated blocks into one or more Chunks, splitting at paragraph
    boundaries (or sentences as last resort) if over budget.
    """
    if not accumulated:
        return []

    chunks: List[Chunk] = []
    current_blocks: List[Block] = []
    current_tokens = 0
    overhead = _estimate_tokens(" > ".join(heading_path) + "\n\n")

    for block in accumulated:
        block_tokens = _estimate_tokens(block.text)
        is_atomic = block.type in {"code", "table", "callout", "list"}

        if current_tokens + block_tokens + overhead <= max_tokens:
            current_blocks.append(block)
            current_tokens += block_tokens
        elif is_atomic:
            if current_blocks:
                chunks.append(Chunk(
                    heading=heading, heading_level=heading_level,
                    heading_path=list(heading_path),
                    blocks=current_blocks,
                    canonical_url=url, title=title,
                ))
            current_blocks = [block]
            current_tokens = block_tokens
        else:
            # Paragraph over budget: split at sentence boundaries
            sub_blocks = _split_block_by_sentences(block, max_tokens - overhead)
            for sub in sub_blocks:
                sub_tokens = _estimate_tokens(sub.text)
                if current_tokens + sub_tokens + overhead <= max_tokens:
                    current_blocks.append(sub)
                    current_tokens += sub_tokens
                else:
                    if current_blocks:
                        chunks.append(Chunk(
                            heading=heading, heading_level=heading_level,
                            heading_path=list(heading_path),
                            blocks=current_blocks,
                            canonical_url=url, title=title,
                        ))
                    current_blocks = [sub]
                    current_tokens = sub_tokens

    if current_blocks:
        chunks.append(Chunk(
            heading=heading, heading_level=heading_level,
            heading_path=list(heading_path),
            blocks=current_blocks,
            canonical_url=url, title=title,
        ))

    return chunks


def chunk(doc: NormalizedDoc, max_tokens: int = MAX_TOKENS) -> List[Chunk]:
    """Convert a NormalizedDoc into semantic Chunks."""
    heading_stack: dict[int, str] = {}
    current_heading = doc.title or "Introduction"
    current_level = 1
    accumulated: List[Block] = []
    chunks: List[Chunk] = []

    def heading_path() -> List[str]:
        return [heading_stack[lvl] for lvl in sorted(heading_stack)]

    def flush() -> None:
        nonlocal accumulated
        chunks.extend(_flush(
            accumulated=accumulated,
            heading=current_heading,
            heading_level=current_level,
            heading_path=heading_path(),
            url=doc.canonical_url,
            title=doc.title,
            max_tokens=max_tokens,
        ))
        accumulated = []

    for block in doc.blocks:
        if block.type == "heading":
            flush()
            level = block.meta.get("level", 2)
            for lvl in list(heading_stack):
                if lvl >= level:
                    del heading_stack[lvl]
            heading_stack[level] = block.text
            current_heading = block.text
            current_level = level
        else:
            accumulated.append(block)

    flush()
    return chunks
