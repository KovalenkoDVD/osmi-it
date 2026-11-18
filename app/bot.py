"""Telegram bot (aiogram 3): answers questions about OSMI IT / OSMI AI from the knowledge base."""

from __future__ import annotations

import asyncio
import html
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, KeyboardButton, LinkPreviewOptions, Message, ReplyKeyboardMarkup
from aiogram.utils.chat_action import ChatActionSender

from app.config import Settings, get_settings
from app.db import Database
from app.embeddings import create_embedder
from app.ingest import run_ingest
from app.llm import create_llm
from app.rag import RAGService
from app.retrieval import Retriever
from app.tracing import Tracing

log = logging.getLogger(__name__)

TG_LIMIT = 4000  # Telegram hard limit is 4096 characters

EXAMPLES = [
    "Чем занимается OSMI IT?",
    "Что такое платформа OSMI AI?",
    "Какие кейсы есть у компании?",
    "Как связаться с компанией?",
    "Какие реквизиты у ООО «ОСМИ-ИТ»?",
    "Какие LLM поддерживает OSMI AI?",
]

# Bot profile shown in Telegram (set automatically on startup, see sync_profile).
BOT_DESCRIPTION = (  # "What can this bot do?" screen, up to 512 characters
    "Демо RAG-ассистента по сайтам OSMI IT и OSMI AI: osmi-it.ru, osmi-ai.ru и docs.osmi-ai.ru.\n\n"
    "Отвечает только по данным сайтов и даёт ссылки на источники: услуги и решения, кейсы, "
    "платформа OSMI AI и её документация, контакты, реквизиты.\n\n"
    "Стек: aiogram 3, PostgreSQL + pgvector, гибридный поиск (вектор + BM25, RRF), Gemini / Claude, Docker.\n"
    "Код: github.com/KovalenkoDVD/osmi-it\n\n"
    "Тестовое задание к вакансии LLM-engineer в OSMI IT."
)
BOT_SHORT_DESCRIPTION = (  # profile "About", up to 120 characters
    "RAG-ассистент по сайтам OSMI IT и OSMI AI: отвечает по данным сайтов со ссылками на источники"
)

HELP_TEXT = (
    "Я отвечаю на вопросы об <b>OSMI IT</b> и платформе <b>OSMI AI</b> по материалам сайтов "
    "osmi-it.ru, osmi-ai.ru и docs.osmi-ai.ru: услуги, кейсы, платформа и её API, интеграции, "
    "награды, новости, контакты, реквизиты.\n\n"
    "Просто напишите вопрос. Я учитываю несколько предыдущих сообщений, поэтому можно уточнять: "
    "«а сколько длится внедрение?».\n\n"
    "Команды:\n"
    "/reset — начать диалог заново\n"
    "/stats — состояние базы знаний\n"
    "/reindex — переобойти сайты и обновить базу (только для админов)"
)


class State:
    """Shared runtime objects (kept outside aiogram DI for simplicity)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.database_url)
        self.embedder = create_embedder(settings)
        self.tracing = Tracing(settings)
        self.retriever = Retriever(
            self.db, self.embedder, top_k=settings.top_k, candidates_k=settings.candidates_k, rrf_k=settings.rrf_k
        )
        self.rag = RAGService(self.db, self.retriever, create_llm(settings), self.tracing, settings.history_turns)
        self.ingest_lock = asyncio.Lock()


def examples_keyboard() -> ReplyKeyboardMarkup:
    buttons = [KeyboardButton(text=text) for text in EXAMPLES]
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True, input_field_placeholder="Ваш вопрос об OSMI IT…")


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + len(paragraph) + 2 > limit:
            parts.append(current)
            current = ""
        while len(paragraph) > limit:  # a single giant paragraph: hard split on a line break
            cut = paragraph.rfind("\n", 0, limit)
            cut = cut if cut > 0 else limit
            parts.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip("\n")
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts


async def reply_html(message: Message, text: str, **kwargs) -> None:
    for part in split_message(text):
        try:
            await message.answer(part, **kwargs)
        except TelegramBadRequest as exc:
            # Broken markup from the model: resend as plain text instead of failing.
            log.warning("HTML rejected by Telegram (%s), sending as plain text", exc)
            await message.answer(html.unescape(part), parse_mode=None, **kwargs)


def build_router(state: State) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(
            "Здравствуйте! Я бот-консультант по компании <b>OSMI IT</b> и платформе ИИ-агентов <b>OSMI AI</b>.\n\n"
            "Спросите что угодно о компании — или выберите пример ниже. /help — подробнее.",
            reply_markup=examples_keyboard(),
        )

    @router.message(Command("help"))
    async def help_(message: Message) -> None:
        await message.answer(HELP_TEXT, reply_markup=examples_keyboard())

    @router.message(Command("reset"))
    async def reset(message: Message) -> None:
        await state.db.clear_history(message.chat.id)
        await message.answer("История диалога очищена. Задайте новый вопрос.")

    @router.message(Command("stats"))
    async def stats(message: Message) -> None:
        s = await state.db.stats()
        llm = state.rag.llm.model if state.rag.llm else "не настроена (режим цитат)"
        building = "\n⏳ Сейчас идёт обновление базы." if state.ingest_lock.locked() else ""
        await message.answer(
            f"<b>База знаний</b>\nДокументов: {s['documents']}\nЧанков: {s['chunks']}\n"
            f"Сайты: {s['hosts'] or '—'}\nОбновлена: {s['updated_at'] or '—'}\n"
            f"Эмбеддинги: {state.embedder.model_name}\nLLM: {llm}{building}"
        )

    @router.message(Command("reindex"))
    async def reindex(message: Message) -> None:
        if message.from_user is None or message.from_user.id not in state.settings.admin_ids:
            await message.answer("Команда доступна только администраторам (ADMIN_IDS).")
            return
        if state.ingest_lock.locked():
            await message.answer("Обновление базы уже идёт.")
            return
        await message.answer("Начинаю обход сайтов и обновление базы знаний. Это займёт несколько минут…")
        report = await ingest(state)
        await message.answer(f"✅ База обновлена: {report}" if report else "❌ Обновление не удалось, см. логи.")

    @router.message(F.text & ~F.text.startswith("/"))
    async def question(message: Message, bot: Bot) -> None:
        text = (message.text or "").strip()
        if len(text) > 2000:
            await message.answer("Слишком длинный вопрос — сформулируйте, пожалуйста, короче (до 2000 символов).")
            return
        if state.ingest_lock.locked() and (await state.db.stats())["chunks"] == 0:
            await message.answer("⏳ База знаний ещё строится (первый запуск), попробуйте через пару минут.")
            return
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            try:
                answer = await state.rag.answer(
                    message.chat.id, text, user_id=message.from_user.id if message.from_user else None
                )
            except Exception:
                log.exception("Failed to answer")
                await message.answer("Произошла ошибка при обработке вопроса. Попробуйте ещё раз чуть позже.")
                return
        await reply_html(message, answer.text)

    @router.message()
    async def other(message: Message) -> None:
        await message.answer("Я понимаю только текстовые вопросы. Напишите, что вы хотите узнать об OSMI IT.")

    return router


async def ingest(state: State, *, full: bool = False):
    async with state.ingest_lock:
        try:
            return await run_ingest(state.db, state.settings, state.embedder, full=full)
        except Exception:
            log.exception("Ingest failed")
            return None


async def sync_profile(bot: Bot) -> None:
    """Fill the bot's description and "About" so nothing has to be typed into @BotFather.

    A description that does not mention OSMI belongs to some other bot (wrong token in .env):
    it is left untouched instead of being overwritten.
    """
    try:
        current = (await bot.get_my_description()).description
        if current and "OSMI" not in current.upper():
            log.warning("Bot already has a foreign description: profile is not changed (check BOT_TOKEN)")
            return
        if current != BOT_DESCRIPTION:
            await bot.set_my_description(description=BOT_DESCRIPTION)
        if (await bot.get_my_short_description()).short_description != BOT_SHORT_DESCRIPTION:
            await bot.set_my_short_description(short_description=BOT_SHORT_DESCRIPTION)
    except Exception:  # profile cosmetics must never stop the bot
        log.exception("Could not update bot profile")


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not settings.bot_token:
        raise SystemExit("BOT_TOKEN is not set (see .env.example)")

    state = State(settings)
    await state.db.connect()
    await state.db.init_schema(state.embedder.model_name, state.embedder.dim)
    await state.embedder.warmup()

    background: set[asyncio.Task] = set()
    if settings.auto_ingest and (await state.db.stats())["chunks"] == 0:
        log.info("Knowledge base is empty: starting the initial crawl in the background")
        task = asyncio.create_task(ingest(state))
        background.add(task)
        task.add_done_callback(background.discard)

    bot = Bot(
        settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML, link_preview=LinkPreviewOptions(is_disabled=True)),
    )
    await sync_profile(bot)
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Начать"),
            BotCommand(command="help", description="Что умеет бот"),
            BotCommand(command="reset", description="Очистить историю диалога"),
            BotCommand(command="stats", description="Состояние базы знаний"),
            BotCommand(command="reindex", description="Обновить базу (админ)"),
        ]
    )
    dp = Dispatcher()
    dp.include_router(build_router(state))
    try:
        log.info("Bot started")
        await dp.start_polling(bot)
    finally:
        state.tracing.flush()
        await state.db.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
