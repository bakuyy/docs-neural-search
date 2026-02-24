'''
This should accomplish a few things:
- the input is a string of HTML or text
    - parse the HTML (beautifulsoup?) and pick the main content container
    - remove the chrome/noise
    - create some sort of structured blocks
    - normalize the text grammatically
- url canonicalization
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urldefrag, urlparse, parse_qsl, urlencode, urlunparse

from bs4 import BeautifulSoup


@dataclass
class Block:
    type: str                
    text: str                
    meta: Dict[str, Any]      


@dataclass
class NormalizedDoc:
    canonical_url: str
    title: str
    blocks: List[Block]
    out_links: List[str]


TRACKING_PARAMS_PREFIXES = ("utm_",)
TRACKING_PARAMS_EXACT = {"gclid", "fbclid"}


def canonicalize_url(raw_url: str) -> str:
    """Make URL stable: remove fragments, drop tracking params, normalize host casing."""
    url_no_frag, _ = urldefrag(raw_url)

    p = urlparse(url_no_frag)
    host = p.netloc.lower()

    # Filter query params
    query_pairs = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        kl = k.lower()
        if any(kl.startswith(pref) for pref in TRACKING_PARAMS_PREFIXES):
            continue
        if kl in TRACKING_PARAMS_EXACT:
            continue
        query_pairs.append((k, v))

    query = urlencode(sorted(query_pairs))
    return urlunparse((p.scheme, host, p.path, p.params, query, ""))


def pick_main_container(soup: BeautifulSoup):
    """
    Try common doc containers first; fallback to body.
    You can add site-specific selectors later (Docusaurus, GitBook, Mintlify, etc.).
    """
    for sel in ["main", "article", "[role=main]", ".markdown", ".content", "#content"]:
        node = soup.select_one(sel)
        if node:
            return node
    return soup.body or soup


def remove_noise(container) -> None:
    """Remove common non-content elements."""
    for tag in container.select("script, style, nav, header, footer, aside"):
        tag.decompose()


    junk_keywords = [
        "sidebar", "toc", "table-of-contents", "breadcrumb", "breadcrumbs",
        "edit", "feedback", "cookie", "banner"
    ]

    to_remove = []
    
    for el in container.find_all(True):
        attrs = " ".join([str(el.get("id","")), " ".join(el.get("class", []))]).lower()
        if any(k in attrs for k in junk_keywords):
            to_remove.append(el)

    for el in to_remove:
        el.decompose()


def dom_to_blocks(container) -> List[Block]:
    """
    Convert the main content DOM into ordered blocks.
    Keep it simple first: headings, paragraphs, lists, code blocks.
    """
    blocks: List[Block] = []

    # create hierarchy
    for el in container.find_all(["title","h1","h2","h3","h4","p","pre","ul","ol","table","blockquote"], recursive=True):
        name = el.name.lower()
        if name == "title":
            text = clean_text(el.get_text(" ", strip=True))
            if text:
                blocks.append(Block(type="title", text=text, meta={}))

        elif name in {"h1","h2","h3","h4"}:
            level = int(name[1])
            text = clean_text(el.get_text(" ", strip=True))
            if text:
                blocks.append(Block(type="heading", text=text, meta={"level": level}))

        elif name == "p":
            text = clean_text(el.get_text(" ", strip=True))
            if text:
                blocks.append(Block(type="p", text=text, meta={}))

        elif name == "pre":
            # Many sites wrap code as <pre><code class="language-python">...</code></pre>
            code = el.get_text("\n", strip=False)
            lang = ""
            code_tag = el.find("code")
            if code_tag:
                classes = " ".join(code_tag.get("class", []))
                # crude language extraction
                for c in classes.split():
                    if c.startswith("language-"):
                        lang = c.replace("language-", "")
                        break
            code = normalize_code(code)
            if code.strip():
                blocks.append(Block(type="code", text=code, meta={"lang": lang}))

        elif name in {"ul","ol"}:
            items = [clean_text(li.get_text(" ", strip=True)) for li in el.find_all("li", recursive=False)]
            items = [it for it in items if it]
            if items:
                blocks.append(Block(type="list", text="\n".join(f"- {it}" for it in items), meta={"ordered": name=="ol"}))

        elif name == "table":
            # MVP option: store table as flattened text (later you can preserve structure)
            text = clean_text(el.get_text(" ", strip=True))
            if text:
                blocks.append(Block(type="table", text=text, meta={}))

        elif name == "blockquote":
            text = clean_text(el.get_text(" ", strip=True))
            if text:
                blocks.append(Block(type="callout", text=text, meta={}))

    return blocks


def extract_out_links(container, base_url: str) -> List[str]:
    links = []
    for a in container.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        abs_url = urljoin(base_url, href)
        links.append(canonicalize_url(abs_url))
    # Dedup preserving order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def clean_text(s: str) -> str:
    return " ".join(s.split())


def normalize_code(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    return "\n".join(lines).strip("\n")


def normalize_html(html: str, url: str) -> NormalizedDoc:
    canonical = canonicalize_url(url)
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)

    container = pick_main_container(soup)
    remove_noise(container)

    blocks = dom_to_blocks(container)
    out_links = extract_out_links(container, base_url=canonical)

    return NormalizedDoc(
        canonical_url=canonical,
        title=title,
        blocks=blocks,
        out_links=out_links,
    )