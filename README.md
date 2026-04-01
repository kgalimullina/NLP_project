# 🎓 LLM-ассистент для абитуриентов (RAG + FAQ + Structured Data)
## 📌 Описание проекта

Проект представляет собой интеллектуального ассистента для абитуриентов, который отвечает на вопросы о поступлении, образовательных программах, стоимости обучения, проходных баллах, количестве мест, требованиях к ЕГЭ и устройстве мегакластеров.

Система построена по принципу RAG (Retrieval Augmented Generation): сначала происходит поиск релевантной информации, затем LLM формирует ответ строго на основе найденного контекста.

Для точных вопросов (стоимость, баллы, ЕГЭ) используется работа с табличными данными.

---

## 🎯 Цель

Снизить нагрузку на сотрудников приёмной комиссии за счёт автоматизации ответов на типовые вопросы абитуриентов и их родителей.

---

## ⚙️ Пайплайн решения
```
Запрос пользователя
    ↓
Модерация (фильтр токсичности)
    ↓
Классификация запроса (FAQ / таблица / общий)
    ↓
Program Retriever (поиск программ)
    ↓
Hybrid Retrieval:
    • BM25 (лексический поиск)
    • Embeddings (Ollama)
    • FAISS (векторный поиск)
    • RRF (объединение результатов)
    ↓
Формирование контекста
    ↓
LLM (GPT-4o-mini)
    ↓
Ответ
```
---

## 🛠️ Стек

LLM:

* GPT-4o-mini (OpenAI API)

Embeddings:

* nomic-embed-text (Ollama)

Retrieval:

* BM25 (rank-bm25)
* FAISS
* Reciprocal Rank Fusion (RRF)

Backend:

* Python
* pandas
* numpy

Интерфейсы:

* Telegram Bot (pyTelegramBotAPI)
* Streamlit

---

## 🚀 Как запустить проект

1. Установить зависимости
   pip install -r requirements.txt

2. Создать файл `.env`
   TG_TOKEN=your_telegram_token
   OPENAI_API_KEY=your_openai_key
   DATA_DIR=.

3. Запуск Telegram-бота
   python bot.py

4. Запуск веб-интерфейса
   streamlit run app.py

---

## 📂 Структура репозитория
```
project/
│
├── bot.py
├── app.py
├── requirements.txt
├── .env.example
│
├── all_program.xlsx
├── Database.xlsx
├── Database-2.xlsx
│
├── ru_abusive_words.txt
├── ru_curse_words.txt
│
└── notebook.ipynb
```





