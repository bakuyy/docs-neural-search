from __future__ import annotations #annotations postpone the evaluation of annotations
import re
from urllib.parse import  urlparse, urlunparse, parse_qsl, urlencode

# define high level filters
ALLOWED_PATHS = re.compile(r"/(docs?|documentation|learn|guide|guides|reference|api|manual)(/|$)", re.I)
BAD_PATHS = re.compile(r"/(blog|pricing|about|careers|jobs|terms|privacy)")
BAD_EXTENSIONS = re.compile(r"(\.pdf|\.docx|\.xlsx|\.pptx|\.doc|\.xls|\.ppt|\.txt|\.csv|\.json|\.xml|\.yaml|\.yml|\.ini|\.conf|\.log|\.err|\.out|\.log\.gz|\.log\.bz2|\.log\.xz|\.log\.lzma|\.log\.tar|\.log\.tar\.gz|\.log\.tar\.bz2|\.log\.tar\.xz|\.log\.tar\.lzma)")    

def canonicalize_url(url: str) -> str:
    """
    Normalize the URL so we do not enqueue duplicates later on
    - we remove fragments, sort query params and normalize the scheme/host casing
    
    examples

    # canonicalized:
    https://example.com/docs?page=2
    https://example.com/docs?page=2#section

    # parsed:
    url = "HTTPS://Example.COM:443/docs/api?b=2&a=1#intro"
    p.scheme   → "https"
    p.netloc   → "Example.COM:443"
    p.path     → "/docs/api"
    p.query    → "b=2&a=1"
    p.fragment → "intro"

    """
    p = urlparse(url) #splits into components above
    scheme = p.scheme.lower() if p.scheme else "https" 
    netloc = p.netloc.lower() #host/port
    path = p.path or "/"

    # remove fragment
    fragment = ""

    # drop common marketing tracking params
    q = [(k,v) for (k,v) in parse_qsl(p.query, keep_blank_values=True)  # parse query params into pairs
    if k.lower() not in {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid"}]

    q.sort() # canonicalize so that params like a=xx and b=xx will always be sorted the same way
    query = urlencode(q)

    # correct format: (scheme, netloc, path, params, query, fragment)
    return urlunparse((scheme, netloc, path, "", query, fragment))


def is_allowed_domain(url:str, allowed_domains: set[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith(f".{d}") for d in allowed_domains)

def is_valid_doc_url(url:str) -> bool:
    p = urlparse(url)
    
    if not p.scheme.startswith("https"):
        return False
    if not ALLOWED_PATHS.search(p.path):
        return False
    if p.path.endswith(BAD_EXTENSIONS):
        return False
    
    lower_path = p.path.lower()
    if any(bad in lower_path for bad in BAD_PATHS):
        return False
    
    return bool(ALLOWED_PATHS.search(lower_path))

