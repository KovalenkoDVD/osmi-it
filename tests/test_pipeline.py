from app.chunking import Chunk, Document, build_documents, chunk_document
from app.crawler import CrawledPage, Robots, normalize_url
from app.db import SearchHit, build_tsquery
from app.extract import Block, parse_html
from app.rag import cited_numbers, group_sources, markdown_to_telegram_html, render_sources
from app.retrieval import rrf_fuse

HTML = """
<html><head><title>Кейсы — OSMI IT</title><meta name="description" content="Наши проекты"></head>
<body>
  <nav><a href="/">Главная</a><a href="/contacts/">Контакты</a></nav>
  <h1>Платформа<br>ИИ-агентов</h1>
  <p>Первый <b>абзац</b> текста.</p>
  <!-- <p>Закомментированный блок</p> -->
  <div style="display: none">Скрытая форма</div>
  <script>var x = 1;</script>
  <h2>Контакты</h2>
  <p><a href="tel:+74951281989">+7 (495) 128-19-89</a></p>
  <a href="https://t.me/osmi_it">Telegram</a>
</body></html>
"""


def test_parse_html_structure():
    page = parse_html(HTML, "https://osmi-it.ru/cases/")
    texts = [b.text for b in page.blocks]
    assert page.title == "Кейсы — OSMI IT"
    assert page.description == "Наши проекты"
    assert Block("Платформа ИИ-агентов", 1) in page.blocks  # <br> does not split the heading
    assert "Первый абзац текста." in texts  # inline tags are glued
    assert not any("Закомментированный" in t or "Скрытая" in t or "var x" in t for t in texts)
    assert any(link.url == "tel:+74951281989" for link in page.links)


def _page(url: str, extra: str, body: str = "") -> CrawledPage:
    html = (
        f"<body><header>Меню сайта</header><h1>{extra}</h1><p>Текст {extra}</p>{body}"
        f"<footer><p>+7 (495) 128-19-89</p><a href='https://t.me/osmi_it'>tg</a></footer></body>"
    )
    return CrawledPage(url, "html", parse_html(html, url))


def test_boilerplate_moved_to_sitewide_document():
    contacts_body = "<p>Телефон для связи:</p><p>+7 (495) 128-19-89</p>"
    pages = [_page("https://osmi-it.ru/contacts/", "Контакты", contacts_body)]
    pages += [_page(f"https://osmi-it.ru/p{i}/", f"Страница {i}") for i in range(5)]
    docs = build_documents(pages)
    page_docs = [d for d in docs if d.kind == "html"]
    sitewide = next(d for d in docs if d.kind == "sitewide")
    contacts = next(d for d in docs if d.kind == "contacts")
    assert all("Меню сайта" not in b.text and b.text != "+7 (495) 128-19-89" for d in page_docs for b in d.blocks)
    assert any(b.text == "Меню сайта" for b in sitewide.blocks)
    # the labelled value on the contacts page is unique text, so it survives boilerplate removal
    contacts_page = next(d for d in page_docs if d.url.endswith("/contacts/"))
    assert any(b.text == "Телефон для связи: +7 (495) 128-19-89" for b in contacts_page.blocks)
    assert any("t.me/osmi_it" in b.text for b in contacts.blocks)


def test_chunking_keeps_heading_path_and_size():
    blocks = [Block("Услуги", 1), Block("ИИ-агенты", 2)] + [Block(f"Предложение номер {i}. " * 8) for i in range(30)]
    doc = Document("https://x.ru/", "OSMI IT", "html", "x.ru", blocks)
    chunks = chunk_document(doc, size=500, overlap=100)
    assert len(chunks) > 5
    assert all(len(c.content) <= 500 for c in chunks)
    assert all(c.heading == "Услуги > ИИ-агенты" for c in chunks)
    assert chunks[0].embed_text.startswith("OSMI IT > Услуги > ИИ-агенты\n")


def test_small_sections_are_merged():
    blocks = [Block("A", 2), Block("один"), Block("B", 2), Block("два")]
    chunks = chunk_document(Document("u", "T", "html", "h", blocks), size=1000)
    assert len(chunks) == 1
    assert "один" in chunks[0].content and "два" in chunks[0].content


def test_robots_and_url_normalisation():
    robots = Robots("User-agent: *\nDisallow: /bitrix/\nDisallow: /*?print=\nDisallow: */index.php$\n")
    assert not robots.allowed("https://osmi-it.ru/bitrix/admin/")
    assert not robots.allowed("https://osmi-it.ru/page/?print=Y")
    assert not robots.allowed("https://osmi-it.ru/a/index.php")
    assert robots.allowed("https://osmi-it.ru/cases/")
    assert normalize_url("http://www.osmi-it.ru/cases/?utm=1#x") == "https://osmi-it.ru/cases/"


def test_tsquery_drops_stopwords():
    assert build_tsquery("Какой ИНН у ООО «ОСМИ-ИТ»?") == "инн | ооо | осми"
    assert build_tsquery("что это?") == ""


def _hit(i: int, url: str = "u") -> SearchHit:
    return SearchHit(i, url, "T", "", f"c{i}")


def test_rrf_prefers_documents_found_by_both_retrievers():
    fused = rrf_fuse([[_hit(1), _hit(2), _hit(3)], [_hit(3), _hit(4)]], k=60, limit=3)
    assert next(h.chunk_id for h in fused) == 3


def test_sources_and_formatting():
    sources = group_sources([_hit(1, "https://a/"), _hit(2, "https://b/"), _hit(3, "https://a/")])
    assert [(s.number, s.url, len(s.hits)) for s in sources] == [(1, "https://a/", 2), (2, "https://b/", 1)]
    assert cited_numbers("Факт [2], ещё [1][2].") == [2, 1]
    html = markdown_to_telegram_html("**ИНН**: 6317139806 <b>\n- пункт\n### Заголовок")
    assert "<b>ИНН</b>" in html and "&lt;b&gt;" in html and "• пункт" in html and "<b>Заголовок</b>" in html
    assert '<a href="https://a/">T</a>' in render_sources(sources)


def test_chunk_embed_text_without_heading():
    assert Chunk(0, "", "текст", "Заголовок").embed_text == "Заголовок\nтекст"


def test_merged_chunk_keeps_sub_headings():
    """Regression: titles of small merged sections used to vanish (e.g. a product name)."""
    blocks = [Block("Решения", 1), Block("Горизонтальные", 2)]
    for name in ["Хаб поддержки", "Планировщик", "Нормализация номенклатуры"]:
        blocks += [Block(name, 3), Block(f"Описание: {name}")]
    blocks += [Block("Вертикальные", 2), Block("Для девелопмента", 3), Block("Продуктовые модули:"), Block("АПО")]
    chunks = chunk_document(Document("u", "T", "html", "h", blocks), size=2000)
    text = "\n".join(c.heading + "\n" + c.content for c in chunks)
    for name in ["Хаб поддержки", "Планировщик", "Нормализация номенклатуры", "Для девелопмента"]:
        assert name in text
    # the whole "Горизонтальные" list stays in one chunk, its titles printed as a tree
    horizontal = next(c for c in chunks if "Хаб поддержки" in c.content)
    assert "## Планировщик\nОписание: Планировщик" in horizontal.content  # markdown heading + its text
    assert "## Нормализация номенклатуры\nОписание: Нормализация номенклатуры" in horizontal.content


def test_section_expansion_adds_neighbours_of_the_same_section():
    import asyncio

    from app.retrieval import Retriever

    def h(i, heading, idx):
        return SearchHit(i, "https://osmi-it.ru/decisions/", "Решения", heading, f"c{i}", 0.0, idx)

    dev = h(2, "Решения > Вертикальные > Для девелопмента", 2)
    industry = h(3, "Решения > Вертикальные", 3)
    horizontal = h(1, "Решения > Горизонтальные", 1)

    class DB:
        async def neighbours(self, ids):
            return [(2, horizontal), (2, industry)] if 2 in ids else []

    retriever = Retriever(DB(), embedder=None)
    result = asyncio.run(retriever._expand_sections([dev]))
    assert [x.chunk_id for x in result] == [2, 3]  # same section added, other section not


def test_lists_and_descriptions_stay_separate():
    html = (
        "<body><h2>Вертикальные</h2><h4>Для девелопмента</h4><p>Единая ИИ-платформа.</p>"
        "<h5>Продуктовые модули</h5><ul><li>ИИ для АПО.</li><li>ИИ-агент для дольщиков.</li></ul></body>"
    )
    page = parse_html(html, "https://x.ru/")
    chunks = chunk_document(Document("u", "T", "html", "x.ru", page.blocks))
    text = chunks[0].content
    assert "Единая ИИ-платформа." in text and "- Единая" not in text  # description is not a list item
    assert "Продуктовые модули\n- ИИ для АПО.\n- ИИ-агент для дольщиков." in text


def test_bot_profile_texts_fit_telegram_limits():
    from app.bot import BOT_DESCRIPTION, BOT_SHORT_DESCRIPTION

    assert len(BOT_DESCRIPTION) <= 512
    assert len(BOT_SHORT_DESCRIPTION) <= 120
