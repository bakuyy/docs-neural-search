from fetcher.fetcher import Fetcher
from fetcher.normalizer import normalize_html
f = Fetcher()

r = f.fetch("https://www.postgresql.org/docs/current/index.html")
normalized = normalize_html(r.html, r.url)
print(normalized)