"""Turn raw HTML / PDF bytes into clean, structured text blocks.

A page becomes a list of ``Block`` objects (heading or paragraph-like text) in
document order. Headings are kept so chunks can carry a breadcrumb such as
"Кейсы > ИИ-ассистент для АО «Заслон»", which noticeably improves retrieval.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urljoin

from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "template",
    "iframe",
    "canvas",
    "video",
    "audio",
    "picture",
    "source",
    "img",
    "input",
    "select",
    "option",
    "textarea",
    "button",
    "form",
    "head",
    "-comment",
    "_comment",
}
BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "dd",
    "details",
    "dialog",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "tbody",
    "thead",
    "tfoot",
    "tr",
    "td",
    "th",
    "ul",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}
logging.getLogger("pypdf").setLevel(logging.ERROR)

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

_WS = re.compile(r"[ \t\r\f\v\u00a0\u200b\u2009\u202f]+")
_COLUMN_GAP = re.compile(r" {3,}")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


@dataclass
class Block:
    text: str
    heading_level: int = 0  # 0 = regular text, 1..6 = heading
    list_item: bool = False  # came from <li>: rendered as "- item" so lists stay lists

    @property
    def is_heading(self) -> bool:
        return self.heading_level > 0


@dataclass
class Link:
    url: str
    text: str


@dataclass
class ParsedPage:
    url: str
    title: str
    description: str = ""
    blocks: list[Block] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)


class _Collector:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._buf: list[str] = []

    def add_text(self, text: str) -> None:
        if text and text.strip():
            self._buf.append(text)

    def flush(self, heading_level: int = 0, list_item: bool = False) -> None:
        if not self._buf:
            return
        text = re.sub(r"\s+", " ", " ".join(self._buf)).strip()
        self._buf = []
        if not text or not re.search(r"\w", text):
            return
        if self.blocks and self.blocks[-1].text == text and self.blocks[-1].heading_level == heading_level:
            return
        self.blocks.append(Block(text=text, heading_level=heading_level, list_item=list_item))


def _is_hidden(node: Node) -> bool:
    attrs = node.attributes
    if "hidden" in attrs or attrs.get("aria-hidden") == "true":
        return True
    style = (attrs.get("style") or "").replace(" ", "").lower()
    return "display:none" in style or "visibility:hidden" in style


def _walk(node: Node, out: _Collector) -> None:
    tag = node.tag
    if tag == "-text":
        out.add_text(node.text(deep=False))
        return
    if tag in SKIP_TAGS or tag.startswith(("-", "_")):
        return
    if _is_hidden(node):
        return

    heading = HEADING_TAGS.get(tag, 0)
    if tag in BLOCK_TAGS:
        out.flush()
    child = node.child
    while child is not None:
        _walk(child, out)
        child = child.next
    if tag in BLOCK_TAGS:
        out.flush(heading_level=heading, list_item=tag == "li")


def parse_html(html: str, url: str) -> ParsedPage:
    tree = HTMLParser(html)

    title = ""
    if tree.css_first("title"):
        title = normalize_ws(tree.css_first("title").text())
    description = ""
    for selector in ('meta[name="description"]', 'meta[property="og:description"]'):
        meta = tree.css_first(selector)
        if meta and meta.attributes.get("content"):
            description = normalize_ws(meta.attributes["content"])
            break

    links: list[Link] = []
    for a in tree.css("a[href]"):
        href = (a.attributes.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "#")):
            continue
        links.append(Link(url=urljoin(url, href), text=normalize_ws(a.text())))

    collector = _Collector()
    root = tree.body or tree.root
    if root is not None:
        _walk(root, collector)
        collector.flush()

    blocks = merge_label_value(collector.blocks)
    return ParsedPage(url=url, title=title, description=description, blocks=blocks, links=links)


def merge_label_value(blocks: list[Block]) -> list[Block]:
    """Glue "Сотрудничество:" + "info@osmi-it.ru" into one line.

    Key/value layouts (contacts, requisites, stats) put the label and the value in
    separate elements; kept apart, the value loses its meaning for retrieval.
    """
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if (
            prev is not None
            and not prev.is_heading
            and not b.is_heading
            and not prev.list_item
            and not b.list_item
            and prev.text.endswith(":")
            and len(prev.text) <= 60
            and len(b.text) <= 200
        ):
            out[-1] = Block(f"{prev.text} {b.text}")
        else:
            out.append(b)
    return out


def parse_pdf(data: bytes, url: str) -> ParsedPage:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    blocks: list[Block] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [ln for ln in text.splitlines() if ln.strip()]
        words = sum(len(ln.split()) for ln in lines)
        if lines and words / len(lines) < 1.5:
            # docx exports put every word on its own line; "layout" mode restores
            # table rows such as "ИНН/КПП | 6317139806/ 502701001".
            text = page.extract_text(extraction_mode="layout") or text
        paragraph: list[str] = []
        for raw in text.splitlines():
            line = normalize_ws(_COLUMN_GAP.sub(" | ", raw.strip()))
            if not line:
                if paragraph:
                    blocks.append(Block("\n".join(paragraph)))
                    paragraph = []
                continue
            paragraph.append(line)
        if paragraph:
            blocks.append(Block("\n".join(paragraph)))
    filename = unquote(url.rsplit("/", 1)[-1])
    return ParsedPage(url=url, title=f"PDF-файл «{filename}»", blocks=blocks)
