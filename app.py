import os
import re
import time
import warnings
from typing import Set

import requests
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", ".")

if not OPENAI_API_KEY:
    raise ValueError("Не найден OPENAI_API_KEY в .env")

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Future Student RANEPA", page_icon="🎓", layout="wide")


# =========================
# Helpers
# =========================
def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if any("nomic" in m for m in models):
            return True, ""
        return False, "Не найдена модель nomic-embed-text"
    except Exception:
        return False, "Ollama не запущен"


def load_toxic_words(data_dir):
    words = set()
    for fname in ["ru_abusive_words.txt", "ru_curse_words.txt"]:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        words.add(w)
    return words


# =========================
# Data loading
# =========================
@st.cache_data
def load_data(data_dir):
    df_programs = pd.read_excel(os.path.join(data_dir, "all_program.xlsx"))
    df_faq = pd.read_excel(os.path.join(data_dir, "Database.xlsx"))
    df_kb = pd.read_excel(os.path.join(data_dir, "Database-2.xlsx"))
    toxic_words = load_toxic_words(data_dir)
    return df_programs, df_faq, df_kb, toxic_words


def build_program_documents(df):
    docs = []
    for _, row in df.iterrows():
        parts = [
            f"Программа: {row['program']}",
            f"Мегакластер: {row['megacluster']}",
            f"Институт: {row['institute']}",
            f"Направление: {row['major']}",
        ]
        if pd.notna(row.get("tracks")) and str(row["tracks"]).strip():
            parts.append(f"Треки: {row['tracks']}")
        parts.append(f"Квалификация: {row.get('qual', '')}")
        parts.append(f"Форма обучения: {row.get('edu_form', '')}")
        parts.append(f"Срок: {row.get('edu_years', '')} лет")
        if pd.notna(row.get("pass_2024")):
            parts.append(f"Проходной балл 2024: {row['pass_2024']}")
        if pd.notna(row.get("budget_2025")):
            parts.append(f"Бюджетных мест 2025: {int(row['budget_2025'])}")
        if pd.notna(row.get("contract_2025")):
            parts.append(f"Платных мест 2025: {int(row['contract_2025'])}")
        if pd.notna(row.get("cost")):
            parts.append(f"Стоимость: {int(row['cost'])} руб./год")
        if pd.notna(row.get("eges_contract")) and str(row["eges_contract"]).strip():
            parts.append(f"ЕГЭ (контракт): {row['eges_contract']}")
        if pd.notna(row.get("eges_budget")) and str(row["eges_budget"]).strip():
            parts.append(f"ЕГЭ (бюджет): {row['eges_budget']}")
        docs.append({
            "source": "program",
            "program_name": row["program"],
            "text": "\n".join(parts),
            "raw": row.to_dict(),
        })
    return docs


def build_faq_documents(df):
    docs = []
    for _, row in df.iterrows():
        q = str(row.get("Question", "")).strip()
        a = str(row.get("Answer", "")).strip()
        if not q:
            continue
        docs.append({
            "source": "faq",
            "question": q,
            "question_type": str(row.get("Question type", "")),
            "text": f"Вопрос: {q}\nОтвет: {a}",
            "answer": a,
        })
    return docs


def build_kb_documents(df):
    docs = []
    for _, row in df.iterrows():
        h = str(row.get("header", "")).strip()
        t = str(row.get("text", "")).strip()
        if not t:
            continue
        docs.append({
            "source": "knowledge_base",
            "header": h,
            "text": f"{h}\n{t}" if h else t,
        })
    return docs


# =========================
# Moderation
# =========================
class Moderator:
    def __init__(self, toxic_words: Set[str]):
        self.toxic_words = toxic_words
        if toxic_words:
            escaped = [re.escape(w) for w in sorted(toxic_words, key=len, reverse=True)]
            self.pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE | re.UNICODE)
        else:
            self.pattern = None

    def check(self, text):
        if not text or not text.strip():
            return False, "Пустой запрос"
        if self.pattern and self.pattern.search(text.lower().strip()):
            return False, "Обнаружена нецензурная или оскорбительная лексика"
        return True, ""

    def get_safe_response(self, reason):
        if "нецензурная" in reason:
            return "К сожалению, запрос содержит некорректную лексику. Переформулируйте, пожалуйста."
        if "Пустой" in reason:
            return "Пожалуйста, введите вопрос."
        return "Не удалось обработать запрос."


# =========================
# Program search
# =========================
class ProgramRetriever:
    _SUFFIXES = [
        "ного", "ной", "ных", "ным", "ому", "ого", "ей", "ой", "ых", "ие", "ые", "ий",
        "ая", "яя", "ое", "ее", "ию", "ую", "ов", "ев", "ам", "ям", "ах", "ях",
        "ке", "ку", "ки", "ка", "ом", "ем", "е", "у", "а", "я", "и", "ы", "о",
    ]

    @classmethod
    def _stem(cls, word):
        if len(word) <= 4:
            return word
        for suf in cls._SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                return word[:-len(suf)]
        return word

    @classmethod
    def _stem_set(cls, text):
        words = re.sub(r"[^\w\sа-яёa-z0-9-]", " ", text.lower()).split()
        return {cls._stem(w) for w in words if len(w) > 2}

    def __init__(self, df):
        self.df = df
        self.names = df["program"].str.lower().tolist()
        self._stems = [self._stem_set(n) for n in self.names]

    def find_program(self, query):
        """Возвращает список подходящих программ."""
        q = query.lower()
        results = []

        # 1) Прямое вхождение названия программы в запрос
        for i, name in enumerate(self.names):
            if name in q:
                results.append(self.df.iloc[i].to_dict())
        if results:
            return results

        # 2) Основа слова из запроса → ищем в названии
        q_words = [w for w in re.sub(r"[^\w\sа-яёa-z0-9-]", " ", q).split() if len(w) > 4]
        for word in sorted(q_words, key=len, reverse=True):
            stem = self._stem(word)
            if len(stem) < 4:
                continue
            for i, name in enumerate(self.names):
                if stem in name:
                    results.append(self.df.iloc[i].to_dict())
            if 0 < len(results) <= 5:
                return results
            results = []

        # 3) Нечёткое по стемам
        q_stems = self._stem_set(q)
        best_score, best = 0, None
        for i, ns in enumerate(self._stems):
            if not ns:
                continue
            common = q_stems & ns
            cov = len(common) / len(ns)
            min_c = 2 if len(ns) > 1 else 1
            if cov >= 0.5 and len(common) >= min_c and cov > best_score:
                best_score = cov
                best = self.df.iloc[i].to_dict()
        return [best] if best else []


# =========================
# Embeddings + retrieval
# =========================
class OllamaEmbedder:
    def __init__(self, base_url="http://localhost:11434", model="nomic-embed-text"):
        self.base_url = base_url
        self.model = model

    def embed(self, texts):
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        r.raise_for_status()
        return np.array(r.json()["embeddings"], dtype=np.float32)


class HybridRetriever:
    def __init__(self, documents, embedder, top_k=5, rrf_k=60, batch_size=50):
        self.documents = documents
        self.embedder = embedder
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.texts = [d["text"] for d in documents]

        self.bm25 = BM25Okapi([self._tok(t) for t in self.texts])

        all_emb = []
        for i in range(0, len(self.texts), batch_size):
            batch = [t[:2000] for t in self.texts[i:i + batch_size]]
            all_emb.append(self.embedder.embed(batch))
        self.embeddings = np.vstack(all_emb)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings /= norms

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    @staticmethod
    def _tok(text):
        return re.sub(r"[^\w\sа-яёa-z0-9]", " ", text.lower()).split()

    def search(self, query, top_k=None):
        top_k = top_k or self.top_k

        scores = self.bm25.get_scores(self._tok(query))
        bm25_idx = np.argsort(scores)[::-1][:top_k * 4].tolist()

        qe = self.embedder.embed([query])
        qe /= np.linalg.norm(qe)
        _, emb_idx = self.index.search(qe, top_k * 4)
        emb_idx = emb_idx[0].tolist()

        rrf = {}
        for rank, i in enumerate(bm25_idx):
            rrf[i] = rrf.get(i, 0) + 1 / (self.rrf_k + rank + 1)
        for rank, i in enumerate(emb_idx):
            rrf[i] = rrf.get(i, 0) + 1 / (self.rrf_k + rank + 1)

        top = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k]
        return [{**self.documents[i], "rrf_score": rrf[i]} for i in top]


# =========================
# LLM helpers
# =========================
SYSTEM_PROMPT = """Ты — виртуальный ассистент приёмной комиссии Президентской академии (РАНХиГС).
Помогай абитуриентам с вопросами о поступлении, программах, стоимости, баллах и ЕГЭ.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе контекста. НЕ придумывай данные.
2. Если информации нет — честно скажи и предложи обратиться в приёмную комиссию.
3. Числа приводи точно из контекста.
4. Будь краток. Отвечай на русском.
5. Если вопрос не про поступление — вежливо сообщи.
6. Не добавляй то, чего нет в контексте.
7. Если в контексте несколько программ — расскажи про КАЖДУЮ. Не пропускай ни одну.
8. Чётко разделяй обязательные предметы ЕГЭ и предметы по выбору.
"""


def format_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        label = {
            "program": "Программа",
            "faq": "FAQ",
            "knowledge_base": "База знаний",
            "program_exact": "Программа (точное)",
            "program_list": "Список программ",
        }.get(d["source"], d["source"])
        parts.append(f"--- Источник {i} ({label}) ---\n{d['text']}")
    return "\n\n".join(parts)


def generate_answer(query, docs, openai_client, model="gpt-4o-mini"):
    ctx = format_context(docs)
    prompt = f"Контекст:\n\n{ctx}\n\n---\nВопрос: {query}\n\nДай точный ответ на основе контекста."
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=700,
    )
    return resp.choices[0].message.content


def classify_query_rule_based(query):
    q = query.lower()
    cost_kw = ["стоимость", "стоит", "цена", "платн", "оплат"]
    ege_kw = ["егэ", "экзамен", "предмет", "балл"]
    places_kw = ["бюджетн", "мест ", "места", "контракт"]
    score_kw = ["проходн", "балл"]

    if any(k in q for k in cost_kw + ege_kw + places_kw + score_kw):
        return "table"

    faq_kw = [
        "что такое", "как подать", "документ", "когда", "можно ли", "отличие",
        "как поступить", "как оплатить", "общежити", "стипенди"
    ]
    if any(k in q for k in faq_kw):
        return "faq"

    return "general"


def handle_analytical_query(query, df):
    q = query.lower()

    if ("дешёв" in q or "дешев" in q or "недорог" in q or "дёшев" in q) and ("программ" in q or "обучен" in q or "стои" in q):
        row = df.loc[df["cost"].idxmin()]
        return f"Самая доступная программа — «{row['program']}» ({row['megacluster']}), стоимость: {int(row['cost'])} руб./год."

    if ("дорог" in q or "максимальн" in q) and ("программ" in q or "обучен" in q or "стои" in q):
        row = df.loc[df["cost"].idxmax()]
        return f"Самая дорогая программа — «{row['program']}» ({row['megacluster']}), стоимость: {int(row['cost'])} руб./год."

    if ("больше всего" in q or "максимум" in q or "наибольш" in q) and "бюджет" in q:
        top = df.nlargest(3, "budget_2025")[["program", "budget_2025"]]
        lines = [f"«{r['program']}» — {int(r['budget_2025'])} мест" for _, r in top.iterrows()]
        return "Программы с наибольшим числом бюджетных мест:\n" + "\n".join(f"• {l}" for l in lines)

    if ("программ" in q or "входят" in q or "входит" in q or "какие" in q) and "мегакластер" in q:
        megaclusters = df["megacluster"].str.lower().unique()
        for mc in megaclusters:
            if mc in q:
                programs = df[df["megacluster"].str.lower() == mc]["program"].tolist()
                lines = [f"• {p}" for p in programs]
                return f"Программы мегакластера «{mc.title()}» ({len(programs)} шт.):\n" + "\n".join(lines)

    return None


DIRECTION_SYNONYMS = {
    "юрист": ["правов", "правовая", "правовой"],
    "экономист": ["экономик", "экономическ", "финанс"],
    "журналист": ["журналист", "медиа", "коммуникац"],
    "дипломат": ["международн", "дипломат", "внешн"],
    "программист": ["информатик", "информацион", "данных", "цифров"],
    "менеджер": ["менеджмент", "управлен"],
}


def expand_query_with_synonyms(query, df_programs):
    q = query.lower()
    additions = []
    for synonym, real_words in DIRECTION_SYNONYMS.items():
        if synonym in q:
            for _, row in df_programs.iterrows():
                name = row["program"].lower()
                major = str(row["major"]).lower()
                if any(rw in name or rw in major for rw in real_words):
                    additions.append(row["program"])
            break
    return additions[:5]


# =========================
# Pipeline
# =========================
class RAGPipeline:
    def __init__(self, moderator, retriever, program_retriever, openai_client, df_programs):
        self.mod = moderator
        self.ret = retriever
        self.prog = program_retriever
        self.client = openai_client
        self.df_programs = df_programs

    def _prog_doc(self, p):
        parts = [
            f"Программа: {p.get('program', '')}",
            f"Мегакластер: {p.get('megacluster', '')}",
            f"Институт: {p.get('institute', '')}",
            f"Направление: {p.get('major', '')}",
            f"Форма: {p.get('edu_form', '')}, {p.get('edu_years', '')} лет",
        ]

        pass_val = p.get("pass_2024")
        if pd.notna(pass_val):
            try:
                parts.append(f"Проходной балл 2024: {int(pass_val)}")
            except (ValueError, TypeError):
                parts.append(f"Проходной балл 2024: {pass_val}")

        budget = p.get("budget_2025")
        if pd.notna(budget):
            parts.append(
                f"Бюджетных мест 2025: {int(budget)}"
                if int(budget) > 0
                else "Бюджетных мест 2025: нет"
            )

        if pd.notna(p.get("contract_2025")) and int(p["contract_2025"]) > 0:
            parts.append(f"Платных мест 2025: {int(p['contract_2025'])}")

        if pd.notna(p.get("cost")):
            parts.append(f"Стоимость: {int(p['cost'])} руб./год")

        if pd.notna(p.get("eges_contract")) and str(p["eges_contract"]).strip():
            parts.append(
                f"ЕГЭ для платного:\n{p['eges_contract']}\n"
                f"ВАЖНО: абитуриент сдаёт обязательные предметы + ОДИН предмет по выбору из списка."
            )

        if pd.notna(p.get("eges_budget")) and str(p["eges_budget"]).strip():
            parts.append(
                f"ЕГЭ для бюджета:\n{p['eges_budget']}\n"
                f"ВАЖНО: абитуриент сдаёт обязательные предметы + ОДИН предмет по выбору из списка."
            )

        return {"source": "program_exact", "text": "\n".join(parts)}

    def process(self, query):
        result = {"answer": "", "status": "ok", "query_type": "", "sources": []}

        safe, reason = self.mod.check(query)
        if not safe:
            result["status"] = "blocked"
            result["answer"] = self.mod.get_safe_response(reason)
            return result

        try:
            analytical = handle_analytical_query(query, self.df_programs)
            if analytical:
                result["query_type"] = "analytical"
                result["answer"] = analytical
                return result

            qtype = classify_query_rule_based(query)
            result["query_type"] = qtype

            docs = []

            # Точный поиск программы (может вернуть несколько)
            found_programs = self.prog.find_program(query)

            if len(found_programs) > 1:
                docs.append({
                    "source": "program_list",
                    "text": "Найдено несколько подходящих программ. Опиши каждую:\n" +
                            "\n".join([p["program"] for p in found_programs[:5]])
                })

            for prog in found_programs[:3]:
                docs.append(self._prog_doc(prog))

            # Расширение через синонимы
            synonym_programs = expand_query_with_synonyms(query, self.df_programs)
            found_names = {p.get("program") for p in found_programs}
            for prog_name in synonym_programs[:2]:
                matches = self.prog.find_program(prog_name)
                for p in matches:
                    if p.get("program") not in found_names:
                        docs.append(self._prog_doc(p))
                        found_names.add(p.get("program"))

            # Гибридный retrieval
            hybrid = self.ret.search(query, top_k=5)
            seen = {d["text"][:100] for d in docs}
            for d in hybrid:
                if d["text"][:100] not in seen:
                    docs.append(d)
                    seen.add(d["text"][:100])

            docs = docs[:5]
            result["sources"] = [{"source": d["source"], "preview": d["text"][:120]} for d in docs]
            result["answer"] = generate_answer(query, docs, self.client)

        except Exception as e:
            result["status"] = "error"
            result["answer"] = f"Ошибка: {e}"

        return result


@st.cache_resource
def build_pipeline(data_dir, openai_api_key):
    df_programs, df_faq, df_kb, toxic_words = load_data(data_dir)
    program_docs = build_program_documents(df_programs)
    faq_docs = build_faq_documents(df_faq)
    kb_docs = build_kb_documents(df_kb)
    all_docs = program_docs + faq_docs + kb_docs

    moderator = Moderator(toxic_words)
    program_retriever = ProgramRetriever(df_programs)
    embedder = OllamaEmbedder()
    retriever = HybridRetriever(all_docs, embedder)
    openai_client = OpenAI(api_key=openai_api_key)
    pipeline = RAGPipeline(moderator, retriever, program_retriever, openai_client, df_programs)

    return {
        "pipeline": pipeline,
        "df_programs": df_programs,
        "df_faq": df_faq,
        "df_kb": df_kb,
        "all_docs_count": len(all_docs),
        "toxic_words_count": len(toxic_words),
    }


# =========================
# UI
# =========================
st.title("🎓 Future Student RANEPA")
st.caption("Ассистент для вопросов о поступлении, программах, стоимости, проходных баллах и ЕГЭ")

if "messages" not in st.session_state:
    st.session_state.messages = []

ok, _ = check_ollama()
if not ok:
    st.error("Сервис временно недоступен. Попробуйте позже.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_app():
    return build_pipeline(DATA_DIR, OPENAI_API_KEY)


with st.spinner("Подготавливаю ассистента..."):
    try:
        app = get_app()
    except Exception as e:
        st.error(f"Ошибка инициализации пайплайна: {e}")
        st.stop()

with st.sidebar:
    st.subheader("О сервисе")
    st.write(f"Программы: {app['df_programs'].shape[0]}")
    st.write(f"FAQ: {app['df_faq'].shape[0]}")
    st.write(f"База знаний: {app['df_kb'].shape[0]}")
    st.write(f"Всего документов: {app['all_docs_count']}")
    st.write(f"Слов в модерации: {app['toxic_words_count']}")
    show_sources = st.checkbox("Показывать источники", value=True)

    if st.button("Очистить чат"):
        st.session_state.messages = []
        st.rerun()

examples = [
    "Что такое мегакластер?",
    "Сколько стоит бизнес-информатика?",
    "Какие ЕГЭ нужны на анализ данных и ИИ?",
    "Есть ли общежитие?",
    "Какая самая дешевая программа?",
]

with st.expander("Примеры вопросов"):
    for ex in examples:
        st.write(f"• {ex}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("Источники"):
                for i, src in enumerate(msg["sources"], 1):
                    st.write(f"{i}. [{src['source']}] {src['preview']}")

user_input = st.chat_input("Напиши вопрос о поступлении")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Ищу ответ..."):
            t0 = time.time()
            result = app["pipeline"].process(user_input)
            dt = time.time() - t0

        if result["status"] == "blocked":
            answer_text = result["answer"]
            st.warning(answer_text)
        elif result["status"] == "error":
            answer_text = result["answer"]
            st.error(answer_text)
        else:
            answer_text = result["answer"]
            st.write(answer_text)
            st.caption(f"Тип запроса: {result['query_type']} · Время ответа: {dt:.1f} сек")

            if show_sources and result.get("sources"):
                with st.expander("Источники"):
                    for i, src in enumerate(result["sources"], 1):
                        st.write(f"{i}. [{src['source']}] {src['preview']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": result.get("sources", [])
    })