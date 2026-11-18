"""Page -> knowledge-base documents -> chunks.

* Lines repeated on most pages of a host (menu, footer, cookie banner, forms) are
  stripped from every page and stored once in a per-host "site-wide" document,
  so contacts from the footer are still searchable but don't pollute every chunk.
* Contact links (tel:, mailto:, social networks) are turned into a separate
  document, because they often exist only as ``href`` attributes.
* Text is split along headings; each chunk carries a "title > section" header.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

from app.crawler import CrawledPage, host_of
from app.extract import Block

BOILERPLATE_SHARE = 0.4  # line present on >= 40% of the host's pages
BOILERPLATE_MIN_PAGES = 5
SOCIAL_MIN_PAGES = 3
REPEATED_LINE_MIN = 3  # same short line 3+ times on a page = button/CTA label
REPEATED_LINE_MAX_LEN = 30  # social links must be site-wide (header/footer), not from one article

SOCIAL_HOSTS = {
    "t.me": "Telegram",
    "telegram.me": "Telegram",
    "vk.com": "ВКонтакте",
    "youtube.com": "YouTube",
    "www.youtube.com": "YouTube",
    "rutube.ru": "Rutube",
    "vc.ru": "vc.ru",
    "habr.com": "Хабр",
    "dzen.ru": "Дзен",
    "max.ru": "MAX",
    "workspace.ru": "Workspace",
    "dprofile.ru": "Dprofile",
    "linkedin.com": "LinkedIn",
    "www.linkedin.com": "LinkedIn",
}


@dataclass
class Document:
    url: str
    title: str
    kind: str
    host: str
    blocks: list[Block]
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.title.encode())
        for chunk in self.chunks:
            h.update(b"\x00" + chunk.embed_text.encode())
        return h.hexdigest()


@dataclass
class Chunk:
    index: int
    heading: str
    content: str
    title: str

    @property
    def embed_text(self) -> str:
        """Text that is embedded / shown to the LLM: content with its context header."""
        header = self.title if not self.heading else f"{self.title} > {self.heading}"
        return f"{header}\n{self.content}"


# --------------------------------------------------------------------------- boilerplate
def find_boilerplate(pages: list[CrawledPage]) -> dict[str, set[str]]:
    by_host: dict[str, list[CrawledPage]] = defaultdict(list)
    for p in pages:
        if p.kind == "html":
            by_host[host_of(p.url)].append(p)
    result: dict[str, set[str]] = {}
    for host, host_pages in by_host.items():
        if len(host_pages) < BOILERPLATE_MIN_PAGES:
            result[host] = set()
            continue
        counter: Counter[str] = Counter()
        for p in host_pages:
            counter.update({b.text for b in p.page.blocks})
        threshold = max(2, int(len(host_pages) * BOILERPLATE_SHARE))
        result[host] = {text for text, n in counter.items() if n >= threshold}
    return result


def _clean_title(title: str, suffix: str = "") -> str:
    title = re.sub(r"\s+", " ", title).strip()
    if suffix and title.endswith(suffix) and len(title) > len(suffix):
        title = title[: -len(suffix)].strip()
    return title or "Без названия"


def common_title_suffix(titles: list[str]) -> str:
    """Site-name tail shared by most titles, e.g. " - Osmi AI - Визуально создает агентов…"."""
    if len(titles) < BOILERPLATE_MIN_PAGES:
        return ""
    counter: Counter[str] = Counter()
    for title in titles:
        for sep in (" - ", " | ", " — "):
            idx = title.find(sep)
            while idx > 0:
                counter[title[idx:]] += 1
                idx = title.find(sep, idx + 1)
    if not counter:
        return ""
    suffix, count = max(counter.items(), key=lambda kv: (kv[1], len(kv[0])))
    return suffix if count >= 0.6 * len(titles) else ""


# --------------------------------------------------------------------------- documents
def build_documents(pages: list[CrawledPage]) -> list[Document]:
    boilerplate = find_boilerplate(pages)
    docs: list[Document] = []
    sitewide_blocks: dict[str, list[Block]] = defaultdict(list)
    sitewide_seen: dict[str, set[str]] = defaultdict(set)
    contacts: dict[str, dict[str, str]] = defaultdict(dict)
    contact_pages: dict[str, Counter[str]] = defaultdict(Counter)
    titles_by_host: dict[str, list[str]] = defaultdict(list)
    for p in pages:
        if p.kind == "html":
            titles_by_host[host_of(p.url)].append(re.sub(r"\s+", " ", p.page.title).strip())
    title_suffix = {host: common_title_suffix(titles) for host, titles in titles_by_host.items()}

    for p in sorted(pages, key=lambda x: x.url):
        host = host_of(p.url)
        common = boilerplate.get(host, set())
        # Short lines repeated on one page are button labels ("Перейти к решению", "Подробнее").
        page_counts = Counter(b.text for b in p.page.blocks if not b.is_heading)
        blocks: list[Block] = []
        for b in p.page.blocks:
            if not b.is_heading and len(b.text) <= REPEATED_LINE_MAX_LEN and page_counts[b.text] >= REPEATED_LINE_MIN:
                continue
            if b.text in common:
                if b.text not in sitewide_seen[host]:
                    sitewide_seen[host].add(b.text)
                    sitewide_blocks[host].append(Block(b.text))
                continue
            blocks.append(b)
        if p.page.description and not any(p.page.description in b.text for b in blocks[:5]):
            blocks.insert(0, Block(p.page.description))
        if blocks:
            docs.append(Document(p.url, _clean_title(p.page.title, title_suffix.get(host, "")), p.kind, host, blocks))

        for link in {link.url: link for link in p.page.links}.values():
            label = _contact_label(link.url, link.text)
            if label:
                value, kind = label
                contact_pages[host][value] += 1
                contacts[host].setdefault(value, kind)

    for host, blocks in sitewide_blocks.items():
        docs.append(
            Document(
                url=f"https://{host}/#site-header-footer",
                title=f"{host}: общая информация из шапки и подвала сайта (меню, контакты)",
                kind="sitewide",
                host=host,
                blocks=blocks,
            )
        )
    for host, items in contacts.items():
        lines = [
            Block(f"{kind}: {value}")
            for value, kind in sorted(items.items(), key=lambda kv: kv[1])
            if kind in ("Телефон", "E-mail") or contact_pages[host][value] >= SOCIAL_MIN_PAGES
        ]
        if not lines:
            continue
        docs.append(
            Document(
                url=f"https://{host}/#contacts",
                title=f"Контакты компании ({host}): телефоны, e-mail, Telegram-канал (телеграм), ВКонтакте, "
                f"YouTube, мессенджеры и соцсети",
                kind="contacts",
                host=host,
                blocks=lines,
            )
        )
    return docs


def _contact_label(url: str, text: str) -> tuple[str, str] | None:
    """Return (value, kind) for contact-like links."""
    if url.startswith("tel:"):
        return (unquote(url[4:]).strip(), "Телефон")
    if url.startswith("mailto:"):
        return (unquote(url[7:]).split("?")[0].strip(), "E-mail")
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return None
    for social_host, name in SOCIAL_HOSTS.items():
        if host == social_host or host.endswith("." + social_host):
            clean = url.split("?")[0].rstrip("/")
            if "/share" in clean:
                return None
            return (clean, name)
    return None


# --------------------------------------------------------------------------- chunking
def _sections(blocks: list[Block]) -> list[tuple[str, list[str]]]:
    """Group blocks under their heading path."""
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    current: list[str] = []

    def heading_path() -> str:
        return " > ".join(t for _, t in stack)

    for b in blocks:
        if b.is_heading:
            if current:
                sections.append((heading_path(), current))
                current = []
            while stack and stack[-1][0] >= b.heading_level:
                stack.pop()
            stack.append((b.heading_level, b.text[:150]))
        else:
            current.append(f"- {b.text}" if b.list_item else b.text)
    if current:
        sections.append((heading_path(), current))
    elif stack:
        sections.append((heading_path(), []))
    return sections


def _split_long_line(line: str, size: int, overlap: int) -> list[str]:
    if len(line) <= size:
        return [line]
    sentences = re.split(r"(?<=[.!?…])\s+", line)
    parts: list[str] = []
    buf = ""
    for s in sentences:
        while len(s) > size:  # pathological: no punctuation at all
            parts.append(s[:size])
            s = s[size - overlap :]
        if buf and len(buf) + len(s) + 1 > size:
            parts.append(buf)
            buf = buf[-overlap:].split(" ", 1)[-1] if overlap else ""
        buf = f"{buf} {s}".strip()
    if buf:
        parts.append(buf)
    return parts


def chunk_document(doc: Document, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    pieces: list[tuple[str, str]] = []  # (heading, text)
    for heading, lines in _sections(doc.blocks):
        if not lines:
            lines = [heading.rsplit(" > ", 1)[-1]] if heading else []
        flat: list[str] = []
        for line in lines:
            flat.extend(_split_long_line(line, size, overlap))
        buf: list[str] = []
        length = 0
        for line in flat:
            if buf and length + len(line) + 1 > size:
                pieces.append((heading, "\n".join(buf)))
                # carry the tail of the previous chunk as overlap
                tail: list[str] = []
                tail_len = 0
                for prev in reversed(buf):
                    if tail_len + len(prev) > overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev) + 1
                buf, length = tail, tail_len
            buf.append(line)
            length += len(line) + 1
        if buf:
            pieces.append((heading, "\n".join(buf)))

    # Merge small neighbouring pieces (short sections like "Быстрый запуск" + 1 sentence).
    # A merged chunk keeps every part's full heading path: the common prefix becomes the
    # chunk heading, the rest of each part's path is written inline, so sub-headings such as
    # "OSMI AI для девелопмента" or a product title are never lost.
    # A new top-level section (e.g. "Горизонтальные ИИ-решения") starts a new chunk unless the
    # current one is tiny, so a list of products is not split between two chunks.
    groups: list[list[tuple[str, str]]] = []
    for heading, text in pieces:
        if groups:
            current = groups[-1]
            fits = _rendered_len([*current, (heading, text)]) <= size
            same_section = _section_key(current[-1][0]) == _section_key(heading)
            if fits and (same_section or _rendered_len(current) < size // 4):
                current.append((heading, text))
                continue
        groups.append([(heading, text)])

    chunks: list[Chunk] = []
    for parts in groups:
        heading, text = _render_group(parts)
        if text.strip():
            chunks.append(Chunk(index=len(chunks), heading=heading, content=text, title=doc.title))
    return chunks


def _section_key(heading: str) -> tuple[str, ...]:
    return tuple(heading.split(" > ")[:2])


def _rendered_len(parts: list[tuple[str, str]]) -> int:
    heading, text = _render_group(parts)
    return len(heading) + len(text)


def _render_group(parts: list[tuple[str, str]]) -> tuple[str, str]:
    """Render parts as Markdown: headings keep their real depth, lists stay "- item" lists.

    The common heading prefix becomes the chunk heading (shown separately); below it only the
    heading levels that change are printed, so the hierarchy "OSMI AI для девелопмента >
    Продуктовые модули > - пункт" is explicit and a description is never confused with a list.
    """
    common = parts[0][0]
    for heading, _ in parts[1:]:
        common = _common_prefix(common, heading)
    base = len(common.split(" > ")) if common else 0
    lines: list[str] = []
    prev_path: list[str] = []
    for heading, text in parts:
        rel = heading[len(common) :].lstrip(" >") if heading.startswith(common) else heading
        path = rel.split(" > ") if rel else []
        same = 0
        while same < min(len(path), len(prev_path)) and path[same] == prev_path[same]:
            same += 1
        for depth, title in enumerate(path[same:], start=base + same + 1):
            lines.append(f"{'#' * min(depth + 1, 6)} {title}")
        prev_path = path
        if path and text == path[-1]:
            continue  # empty section: the heading line is all there is
        lines.append(text)
    return common, "\n".join(lines)


def _common_prefix(a: str, b: str) -> str:
    pa, pb = a.split(" > "), b.split(" > ")
    out = []
    for x, y in zip(pa, pb, strict=False):
        if x != y:
            break
        out.append(x)
    return " > ".join(out)


def build_chunks(pages: list[CrawledPage], size: int = 1200, overlap: int = 200) -> list[Document]:
    docs = build_documents(pages)
    for doc in docs:
        doc.chunks = chunk_document(doc, size, overlap)
    return [d for d in docs if d.chunks]
