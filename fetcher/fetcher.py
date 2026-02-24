from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import time
import random
import gzip
from io import BytesIO

import requests


@dataclass
class FetchResult:
    ok: bool
    url: str             
    final_url: str           # deals with redirects
    status_code: int
    content_type: str
    headers: Dict[str, str]
    fetched_at: float

    # payload (one of these will usually be set when ok=True)
    body_bytes: Optional[bytes] = None
    html: Optional[str] = None

    # error info
    error: Optional[str] = None


class SimplePoliteness:
    """
    Per-host crawl delay controller.
    Ensures we don't hit the same host too frequently.
    """
    def __init__(self, min_delay_s: float = 0.7, jitter_s: float = 0.2):
        self.min_delay_s = float(min_delay_s)
        self.jitter_s = float(jitter_s)
        self._last_fetch_at: Dict[str, float] = {}

    def wait(self, url: str) -> None:
        host = urlparse(url).netloc.lower()
        now = time.time()
        last = self._last_fetch_at.get(host)
        if last is not None:
            elapsed = now - last
            target = self.min_delay_s + random.random() * self.jitter_s
            if elapsed < target:
                time.sleep(target - elapsed)
        self._last_fetch_at[host] = time.time()


class Fetcher:
    """
    Fetches a URL safely for a crawler.
    - Always fetches bytes first.
    - Validates status/size/content-type.
    - Decompresses gzip if needed.
    - Decodes HTML into a string when appropriate.
    """
    def __init__(
        self,
        user_agent: str = "neural-search-crawler/0.1",
        timeout: Tuple[float, float] = (5.0, 20.0),  # (connect, read)
        max_bytes: int = 6 * 1024 * 1024,           # 6 MB
        max_retries: int = 2,
        politeness: Optional[SimplePoliteness] = None,
    ):
        self.timeout = timeout
        self.max_bytes = int(max_bytes)
        self.max_retries = int(max_retries)
        self.politeness = politeness or SimplePoliteness()

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
        })

    def fetch(self, url: str) -> FetchResult:
        # must respect politeness before each request attempt
        last_err: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            self.politeness.wait(url)

            try:
                resp = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                    stream=True,  # allows us to enforce max_bytes while reading
                )

                status = int(resp.status_code)
                final_url = str(resp.url)
                headers = {k: v for k, v in resp.headers.items()}
                ctype = headers.get("Content-Type", "").lower()

                # Handle obvious retry-worthy statuses
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    # basic backoff
                    time.sleep((2 ** attempt) * 0.5 + random.random() * 0.25)
                    continue

                # If not OK, return (no parsing)
                if status < 200 or status >= 300:
                    return FetchResult(
                        ok=False,
                        url=url,
                        final_url=final_url,
                        status_code=status,
                        content_type=ctype,
                        headers=headers,
                        fetched_at=time.time(),
                        error=f"HTTP {status}",
                    )

                # Read bytes up to max_bytes
                raw = self._read_limited(resp, self.max_bytes)

                # Decompress if needed
                raw = self._maybe_decompress(raw, headers)

                # classify + decode HTML when appropriate
                html = None
                if self._looks_like_html(ctype, raw):
                    html = self._decode_html(raw, headers)

                return FetchResult(
                    ok=True,
                    url=url,
                    final_url=final_url,
                    status_code=status,
                    content_type=ctype,
                    headers=headers,
                    fetched_at=time.time(),
                    body_bytes=raw,
                    html=html,
                )

            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    time.sleep((2 ** attempt) * 0.5 + random.random() * 0.25)
                    continue
                break
            except ValueError as e:
                # e.g., size limit hit
                return FetchResult(
                    ok=False,
                    url=url,
                    final_url=url,
                    status_code=0,
                    content_type="",
                    headers={},
                    fetched_at=time.time(),
                    error=str(e),
                )

        return FetchResult(
            ok=False,
            url=url,
            final_url=url,
            status_code=0,
            content_type="",
            headers={},
            fetched_at=time.time(),
            error=last_err or "Unknown fetch error",
        )

    @staticmethod
    def _read_limited(resp: requests.Response, max_bytes: int) -> bytes:
        total = 0
        chunks = []
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Response too large (> {max_bytes} bytes)")
            chunks.append(chunk)
        return b"".join(chunks)
    @staticmethod
    def _maybe_decompress(data: bytes, headers: Dict[str, str]) -> bytes:
        # Only decompress if the payload is actually gzip-compressed.
        # requests often auto-decompresses, but may keep the gzip header.
        if len(data) >= 2 and data[:2] == b"\x1f\x8b":
            try:
                return gzip.GzipFile(fileobj=BytesIO(data)).read()
            except BadGzipFile:
                # If it claims gzip but isn't, just return raw.
                return data
        return data

    @staticmethod
    def _looks_like_html(content_type: str, data: bytes) -> bool:
        if "text/html" in content_type or "application/xhtml+xml" in content_type:
            return True
        # Some servers mislabel. Use a small sniff.
        head = data[:512].lstrip().lower()
        return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<head" in head

    @staticmethod
    def _decode_html(data: bytes, headers: Dict[str, str]) -> str:
        # Respect requests' apparent encoding if provided; else fall back to utf-8.
        # You can improve this later by parsing <meta charset=...>.
        charset = "utf-8"
        ctype = headers.get("Content-Type", "")
        if "charset=" in ctype.lower():
            charset = ctype.lower().split("charset=", 1)[1].split(";", 1)[0].strip()

        try:
            return data.decode(charset, errors="replace")
        except LookupError:
            return data.decode("utf-8", errors="replace")


#test
if __name__ == "__main__":
    f = Fetcher(user_agent="sophie-neural-doc-crawler/0.1")

    r = f.fetch("https://www.postgresql.org/docs/current/index.html")
    print("OK:", r.ok, "status:", r.status_code, "ctype:", r.content_type)
    if r.html:
        print("HTML chars:", len(r.html))
    else:
        print("Not HTML (maybe XML sitemap or other). bytes:", 0 if r.body_bytes is None else len(r.body_bytes))
