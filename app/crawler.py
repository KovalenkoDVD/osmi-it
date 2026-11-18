"""Polite async BFS crawler limited to the configured hosts.

Seeds: start URLs + every URL listed in each host's sitemap(s). Follows internal
links, downloads linked PDFs (presentations, requisites) and respects the simple
``Disallow`` rules from robots.txt.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx

from app.extract import ParsedPage, parse_html, parse_pdf

log = logging.getLogger(__name__)

SKIP_EXTENSIONS = re.compile(
    r"\.(?:jpe?g|png|gif|webp|svg|ico|bmp|tiff?|mp4|webm|mov|avi|mp3|wav|zip|rar|7z|gz|tar|"
    r"docx?|xlsx?|pptx?|css|js|json|xml|woff2?|ttf|eot|otf|php)$",
    re.I,
)
MAX_PDF_BYTES = 25 * 1024 * 1024


@dataclass
class CrawledPage:
    url: str
    kind: str  # "html" | "pdf"
    page: ParsedPage


def host_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def normalize_url(url: str) -> str:
    """Drop query/fragment, unify scheme/host so the same page is visited once."""
    parts = urlsplit(url.strip())
    path = unquote(parts.path) or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path.endswith("/index.php"):
        path = path[: -len("index.php")]
    return urlunsplit(("https", host_of(url), path, "", ""))


class Robots:
    """Tiny robots.txt matcher (``User-agent: *`` Disallow rules with ``*``/``$``)."""

    def __init__(self, text: str = "") -> None:
        self.rules: list[re.Pattern[str]] = []
        applies = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (p.strip() for p in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                applies = value == "*"
            elif key == "disallow" and applies and value:
                pattern = re.escape(value).replace(r"\*", ".*")
                if pattern.endswith(r"\$"):
                    pattern = pattern[:-2] + "$"
                self.rules.append(re.compile(pattern))

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        target = unquote(parts.path) + (f"?{parts.query}" if parts.query else "")
        return not any(rule.match(target) for rule in self.rules)


class Crawler:
    def __init__(
        self,
        start_urls: list[str],
        allowed_hosts: list[str],
        *,
        max_pages: int = 600,
        concurrency: int = 6,
        timeout: float = 30.0,
        user_agent: str = "osmi-rag-bot",
    ) -> None:
        self.start_urls = start_urls
        self.allowed_hosts = {h.lower().removeprefix("www.") for h in allowed_hosts}
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent, "Accept-Language": "ru,en;q=0.8"},
        )
        self.robots: dict[str, Robots] = {}
        self.seen: set[str] = set()
        self.results: dict[str, CrawledPage] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    # ---------------------------------------------------------------- helpers
    def in_scope(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return False
        if host_of(url) not in self.allowed_hosts:
            return False
        path = unquote(parts.path)
        if SKIP_EXTENSIONS.search(path):
            return False
        robots = self.robots.get(host_of(url))
        return robots is None or robots.allowed(url)

    def enqueue(self, url: str) -> None:
        if not self.in_scope(url):
            return
        norm = normalize_url(url)
        if norm in self.seen or len(self.seen) >= self.max_pages:
            return
        self.seen.add(norm)
        self.queue.put_nowait(norm)

    async def _get(self, url: str) -> httpx.Response | None:
        for attempt in range(3):
            try:
                resp = await self.client.get(url)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError("server error", request=resp.request, response=resp)
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if attempt == 2:
                    log.warning("Failed to fetch %s: %s", url, exc)
                    return None
                await asyncio.sleep(1.5 * (attempt + 1))
        return None

    async def _load_robots(self, host: str) -> None:
        resp = await self._get(f"https://{host}/robots.txt")
        self.robots[host] = Robots(resp.text if resp is not None and resp.status_code == 200 else "")

    async def _sitemap_urls(self, url: str, depth: int = 0) -> list[str]:
        resp = await self._get(url)
        if resp is None or resp.status_code != 200 or depth > 2:
            return []
        locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", resp.text)
        if "<sitemapindex" in resp.text:
            nested = await asyncio.gather(*(self._sitemap_urls(loc, depth + 1) for loc in locs))
            return [u for sub in nested for u in sub]
        return locs

    # ------------------------------------------------------------------- core
    async def _process(self, url: str) -> None:
        resp = await self._get(url)
        if resp is None or resp.status_code != 200:
            if resp is not None:
                log.info("Skip %s: HTTP %s", url, resp.status_code)
            return
        final_url = normalize_url(str(resp.url))
        if host_of(final_url) not in self.allowed_hosts:
            return
        if final_url != url and final_url in self.results:
            return  # redirect to a page we already have

        ctype = resp.headers.get("content-type", "").lower()
        if "pdf" in ctype or final_url.lower().endswith(".pdf"):
            if len(resp.content) > MAX_PDF_BYTES:
                return
            try:
                page = parse_pdf(resp.content, final_url)
            except Exception as exc:  # broken PDFs should not stop the crawl
                log.warning("PDF parse failed %s: %s", final_url, exc)
                return
            self.results[final_url] = CrawledPage(final_url, "pdf", page)
            return
        if "html" not in ctype:
            return

        page = parse_html(resp.text, final_url)
        if "404" in page.title or "не найдена" in page.title.lower():
            return
        self.results[final_url] = CrawledPage(final_url, "html", page)
        for link in page.links:
            self.enqueue(link.url)

    async def _worker(self) -> None:
        while True:
            url = await self.queue.get()
            try:
                await self._process(url)
            except Exception:
                log.exception("Unexpected error while crawling %s", url)
            finally:
                self.queue.task_done()

    async def crawl(self) -> list[CrawledPage]:
        hosts = {host_of(u) for u in self.start_urls} | self.allowed_hosts
        await asyncio.gather(*(self._load_robots(h) for h in hosts))

        for url in self.start_urls:
            self.enqueue(url)
        sitemap_lists = await asyncio.gather(*(self._sitemap_urls(f"https://{h}/sitemap.xml") for h in hosts))
        for urls in sitemap_lists:
            for url in urls:
                self.enqueue(url)

        workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]
        try:
            await self.queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await self.client.aclose()
        log.info("Crawled %d documents (%d URLs discovered)", len(self.results), len(self.seen))
        return list(self.results.values())
