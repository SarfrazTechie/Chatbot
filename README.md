# 🤖 FAQ Chatbot — NLP-Powered Question Answering System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-76B900?style=for-the-badge&logo=python&logoColor=white)](https://nltk.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)

**A production-ready FAQ chatbot powered by TF-IDF vectorization and cosine similarity,  
with a clean Streamlit web UI, CLI mode, and full query analytics logging.**

[🚀 Quick Start](#-quick-start) · [🧠 How It Works](#-how-it-works) · [📁 Dataset](#-dataset) · [📊 Analytics](#-query-logging--analytics) · [🛣️ Roadmap](#-upgrade-roadmap)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔤 **Text Preprocessing** | Tokenization → stopword removal → WordNet lemmatization (NLTK) |
| 📊 **TF-IDF Vectorization** | Bigram-aware term weighting with configurable vocabulary |
| 🎯 **Cosine Similarity** | Semantic matching with confidence scoring (0–100%) |
| 🌐 **Web UI** | Responsive Streamlit chat interface with history |
| 📉 **Confidence Bar** | Visual score bar for every answer |
| ⚠️ **Fallback Handling** | Graceful "no match found" with suggested alternatives |
| 📝 **Query Logging** | Timestamped CSV log of every query for gap analysis |
| 📋 **CLI Mode** | Terminal interface for fast testing and scripting |
| 🧪 **Unit Tests** | Full pytest suite covering all core components |
| 🔧 **Configurable** | Threshold, n-gram range, vocab size — all tunable |

---

## 🗂️ Project Structure

```
faq_chatbot/
├── app.py                    # Streamlit web application (main UI)
├── chatbot_cli.py            # Command-line interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/
│   └── faqs.json             # FAQ dataset (20 sample Q&A pairs, 6 categories)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py       # NLP pipeline: clean → tokenize → lemmatize
│   ├── matcher.py            # TF-IDF + cosine similarity matching engine
│   └── logger.py             # CSV query logger with timestamps
│
├── logs/
│   └── query_log.csv         # Auto-generated query log (add to .gitignore)
│
└── tests/
    ├── __init__.py
    └── test_chatbot.py       # pytest unit tests
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/faq-chatbot.git
cd faq-chatbot
```

### Step 2 — Create a virtual environment *(recommended)*

```bash
# Using venv
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

```bash
# Alternatively, using conda
conda create -n faq-chatbot python=3.10
conda activate faq-chatbot
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — NLTK resources

NLTK data (tokenizers, stopwords, WordNet lemmatizer) is **auto-downloaded on first run**. To pre-download manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🚀 Quick Start

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**

> The chatbot loads your FAQ dataset, builds the TF-IDF index, and is ready to answer questions in seconds.

### Command-Line Interface

```bash
python chatbot_cli.py
```

Type a question and press Enter. Type `exit` or `quit` to stop.

```
> How do I reset my password?
Answer: To reset your password, visit the login page and click "Forgot Password"...
Confidence: 87%   |   Category: Account
```

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=utils --cov-report=term-missing
```

---

## 🧠 How It Works

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────┐
│  1. Text Preprocessing  (utils/preprocessor.py)    │
│     ├─ Lowercase & strip punctuation               │
│     ├─ Tokenize with NLTK word_tokenize            │
│     ├─ Remove English stopwords                    │
│     └─ Lemmatize tokens with WordNetLemmatizer     │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│  2. TF-IDF Vectorization  (utils/matcher.py)       │
│     ├─ Transform preprocessed query → vector       │
│     ├─ Unigrams + bigrams (ngram_range=(1,2))      │
│     └─ Vocabulary capped at max_features=5000      │
└────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────┐
│  3. Cosine Similarity Matching                     │
│     ├─ Compare query vector vs. all FAQ vectors    │
│     └─ Rank by similarity score (0.0 to 1.0)      │
└────────────────────────────────────────────────────┘
    │
    ├── Score ≥ threshold (0.2) ──▶ Return best match + confidence
    │
    └── Score < threshold ────────▶ Return fallback response
```

### Why TF-IDF + Cosine Similarity?

**TF-IDF (Term Frequency–Inverse Document Frequency)** weighs terms by how meaningful they are. Words like *"the"* or *"is"* appear everywhere and get low weight. Domain-specific terms like *"invoice"* or *"subscription"* are rare and get high weight — making matches far more accurate.

**Cosine Similarity** measures the angle between two document vectors, not their magnitude. This makes it robust to differences in query length: a 3-word question and a 15-word question about the same topic will still score high similarity.

Together, they form a fast, interpretable baseline that outperforms simple keyword search with zero deep-learning overhead.

---

## 📁 Dataset

The `data/faqs.json` file ships with **20 handcrafted Q&A pairs** across 6 categories:

| Category | Count | Example Question |
|---|---|---|
| Account | 4 | *"How do I reset my password?"* |
| Billing | 4 | *"How can I cancel my subscription?"* |
| Technical | 4 | *"Why is the app running slowly?"* |
| Privacy | 2 | *"How is my data stored?"* |
| Features | 3 | *"Does the app support dark mode?"* |
| Support | 3 | *"How do I contact customer support?"* |

### Adding your own FAQs

Edit `data/faqs.json` and add entries following this schema:

```json
{
  "id": 21,
  "category": "Shipping",
  "question": "How long does standard delivery take?",
  "answer": "Standard delivery takes 3–5 business days. You will receive a tracking number by email once your order has been dispatched.",
  "tags": ["shipping", "delivery", "order"]
}
```

> **Tip:** The more question variants you add per topic, the better the matching accuracy. Consider adding 2–3 phrasings of the same question.

---

## 📊 Query Logging & Analytics

Every user query is automatically logged to `logs/query_log.csv`:

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | ISO 8601 date and time of query |
| `user_query` | string | Raw, unprocessed user input |
| `matched_question` | string | Best-matched FAQ question |
| `confidence` | float | Similarity score as percentage (0–100) |
| `category` | string | FAQ category of the matched answer |
| `response_time_ms` | int | Query processing time in milliseconds |
| `is_fallback` | boolean | `True` if no match exceeded the threshold |

### Using the log for improvement

```python
import pandas as pd

df = pd.read_csv('logs/query_log.csv')

# Identify gaps — queries that fell back with no match
gaps = df[df['is_fallback'] == True]['user_query']
print(gaps.value_counts().head(10))

# Find low-confidence answers worth reviewing
low_conf = df[df['confidence'] < 50]
print(low_conf[['user_query', 'matched_question', 'confidence']])
```

> Add `logs/` to your `.gitignore` to avoid committing user data.

---

## 🔧 Configuration

All key parameters are in `utils/matcher.py`:

| Parameter | Default | Description |
|---|---|---|
| `confidence_threshold` | `0.2` | Minimum cosine score to return a match (0.0–1.0) |
| `ngram_range` | `(1, 2)` | Include unigrams and bigrams in the TF-IDF model |
| `max_features` | `5000` | Maximum vocabulary size |
| `use_lemmatization` | `True` | Apply WordNet lemmatization during preprocessing |
| `fallback_message` | `"I'm not sure..."` | Response when no match is found |

**Tuning tips:**

- Raise `confidence_threshold` (e.g. `0.35`) to reduce irrelevant answers at the cost of more fallbacks.
- Lower it (e.g. `0.1`) to always return *something*, even for loosely-related queries.
- Add `(1, 3)` trigrams if your FAQ questions are long and descriptive.

---

## 🧪 Testing

The test suite in `tests/test_chatbot.py` covers:

- **Preprocessor** — tokenization, stopword removal, lemmatization output
- **Matcher** — correct match returned for exact and paraphrased questions
- **Fallback** — `is_fallback=True` triggered for out-of-domain queries
- **Logger** — CSV row written with correct columns and types
- **Edge cases** — empty string input, single character, all-stopword query

```bash
python -m pytest tests/ -v --tb=short
```

Expected output:

```
tests/test_chatbot.py::test_preprocessor_basic          PASSED
tests/test_chatbot.py::test_preprocessor_lemmatization  PASSED
tests/test_chatbot.py::test_matcher_exact_match         PASSED
tests/test_chatbot.py::test_matcher_paraphrase          PASSED
tests/test_chatbot.py::test_matcher_fallback            PASSED
tests/test_chatbot.py::test_logger_writes_row           PASSED
tests/test_chatbot.py::test_empty_query_handled         PASSED
```

---

## 🛣️ Upgrade Roadmap

This project is designed as a foundation. Here's a clear path from beginner → production:

| Level | Upgrade | Expected Impact |
|---|---|---|
| 🟡 Intermediate | **BM25** ranking (rank_bm25) instead of TF-IDF | Better keyword ranking, faster retrieval |
| 🟡 Intermediate | **Spell correction** (pyspellchecker or symspell) | Handles typos gracefully |
| 🟡 Intermediate | **Redis caching** for repeat queries | Sub-millisecond response for popular FAQs |
| 🟠 Advanced | **Sentence-BERT** embeddings (sentence-transformers) | True semantic understanding, paraphrase-robust |
| 🟠 Advanced | **Cross-encoder reranking** on top-N candidates | High precision without full re-embedding |
| 🟠 Advanced | **Multi-language support** (mBART / LaBSE) | Serve a global user base |
| 🔴 Expert | **Fine-tune DistilBERT** on your FAQ pairs | Near-human accuracy on domain-specific queries |
| 🔴 Expert | **Intent classification** layer | Route multi-intent queries correctly |
| 🔴 Expert | **Retrieval-Augmented Generation** (RAG) | Generate fluent answers grounded in your docs |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────┐
│              User Interface              │
│   Streamlit Web App  │  CLI Terminal     │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│            Chatbot Core                  │
│  preprocessor.py  │  matcher.py         │
│  (NLTK pipeline)  │  (TF-IDF engine)    │
└──────────────────────┬──────────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
┌─────────────────┐    ┌─────────────────────┐
│  data/faqs.json │    │  logs/query_log.csv  │
│  (FAQ dataset)  │    │  (Analytics logging) │
└─────────────────┘    └─────────────────────┘
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/add-bm25`
3. **Write tests** for your changes in `tests/`
4. **Commit** your changes: `git commit -m 'feat: add BM25 ranking support'`
5. **Push** to the branch: `git push origin feature/add-bm25`
6. **Open a Pull Request** with a clear description

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## 📋 Requirements

```
streamlit>=1.28.0
nltk>=3.8.1
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

Install everything at once:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the **MIT License** — free to use for personal, academic, and commercial purposes. See [LICENSE](LICENSE) for details.

---

## 👤 Author

Built as an internship project demonstrating practical NLP engineering skills — from text preprocessing through TF-IDF matching to a production-ready Streamlit interface.

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

*Questions or suggestions? Open an issue or start a discussion.*

</div>
