# for context, a sitemap is a machine-readable list of URLs that a website explicilty publishes for crawlers
from __future__ import annotations
import gzip
import io
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

HEADERS = {"User-Agent": ""}

def candidate_sitemap_urls(start_url: str) -> list[str]:
    """
    Try to find sitemap URLs in the start URL -> check common paths
    """
    base = start_url.split("/", 3)[:3]
    base = "/".join(base) + "/"
    return [
        urljoin(base, "sitemap.xml"),
        urljoin(base, "sitemap_index.xml"),
        urljoin(base, "sitemap/sitemap.xml"),
    ]

def fetch_bytes(url:str, timeout:float = 10.0) -> tuple[int, bytes, str]:
    """ 
    given a URL, we will perform a GET request and return the status code, content, and content type in raw bytes (with min metadata)
    specifically: it returns:
        - HTTP status code
        - raw bytes of the content
        - content type 
    """
    
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    ctype = r.headers.get("Content-Type", "").lower()
    return r.status_code, r.content, ctype

def parse_sitemap(cml_bytes: bytes) -> list[str]:
    """
    supports <urlset> and <sitemapindex> formats
    returns list of urls or nested sitemap urls

    > check first if its gzip, if so wrap bytes in file-like object and decompress into raw xml bytes
    """
    # Handle gzipped sitemaps
    if xml_bytes[:2] == b"\x1f\x8b":
        xml_bytes = gzip.GzipFile(fileobj=io.BytesIO(xml_bytes)).read()

    root = ET.fromstring(xml_bytes)
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    urls = []
    if root.tag.endswith("urlset"):
        for url_el in root.findall(f".//{ns}url/{ns}loc"):
            if url_el.text:
                urls.append(url_el.text.strip())
    elif root.tag.endswith("sitemapindex"):
        for loc_el in root.findall(f".//{ns}sitemap/{ns}loc"):
            if loc_el.text:
                urls.append(loc_el.text.strip())
    return urls

def expand_sitemaps( seed_sitemaps: list[str], max_depth: int = 25) -> list[str]:
    """
    BFS over sitemap idnex -> sitemap -> urls -> returns all discovered page URLs
    """

    pages: list[str] = []
    queue = list(seed_sitemaps)
    seen = set()

    while queue and len(seen) < max_depth:
        sm = queue.pop(0)
        if sm in seen:
            continue
        seen.add(sm)

        try:
            status, content, _ = fetch_bytes(sm)
            if status != 200:
                continue
            items = parse_sitemap(content)
        except Exception:
            continue

        if items and items[0].endswith(".xml", ".xml.gz", ".xml.bz2", ".xml.lzma", ".xml.tar", ".xml.tar.gz", ".xml.tar.bz2", ".xml.tar.lzma"):
            queue.extend(items)
        else:
            pages.extend(items)

    return pages