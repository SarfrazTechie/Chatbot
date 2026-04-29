# 💬 FAQ Chatbot — NLP-Powered Question Answering System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://nltk.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A professional, beginner-friendly FAQ chatbot that uses **NLP text preprocessing**, **TF-IDF vectorization**, and **cosine similarity** to match user questions to the most relevant answers from a custom FAQ dataset. Built with a clean Streamlit UI and full query logging.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔤 Text Preprocessing | Tokenization, stopword removal, lemmatization via NLTK |
| 📊 TF-IDF Vectorization | Bigram-aware term weighting using scikit-learn |
| 🎯 Cosine Similarity | Semantic matching with confidence scoring |
| 🌐 Web UI | Interactive Streamlit chat interface |
| 📉 Confidence Score | Visual score bar for each answer |
| ⚠️ Fallback Response | Graceful handling when no match is found |
| 📝 Query Logging | CSV logging of all queries with timestamps |
| 📋 CLI Mode | Terminal interface for quick testing |
| 🧪 Unit Tests | Pytest suite covering all core components |

---

## 🗂️ Project Structure

```
faq_chatbot/
├── app.py                  # Streamlit web application
├── chatbot_cli.py          # Command-line interface
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/
│   └── faqs.json           # FAQ dataset (20 sample Q&A pairs)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py     # NLP text preprocessing pipeline
│   ├── matcher.py          # TF-IDF + cosine similarity engine
│   └── logger.py           # CSV query logger
│
├── logs/
│   └── query_log.csv       # Auto-generated query log (gitignore this)
│
└── tests/
    └── test_chatbot.py     # Unit tests (pytest)
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/faq-chatbot.git
cd faq-chatbot
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. NLTK resources are auto-downloaded on first run.

---

## 🚀 Usage

### Web Interface (Streamlit)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

### Command Line Interface
```bash
python chatbot_cli.py
```

### Run Tests
```bash
python -m pytest tests/ -v
```

---

## 🧠 How It Works

```
User Query
    │
    ▼
┌─────────────────────┐
│  Text Preprocessing  │  ← Clean → Tokenize → Remove Stopwords → Lemmatize
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  TF-IDF Vectorizer  │  ← Transform query into numerical vector
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Cosine Similarity  │  ← Compare against all FAQ question vectors
└─────────────────────┘
    │
    ├─── Score ≥ threshold → Return best match + confidence score
    │
    └─── Score < threshold → Return fallback response
```

**Why TF-IDF + Cosine Similarity?**
- **TF-IDF** weighs terms by how important they are in a document relative to the full corpus — common words like "the" get low weight, domain-specific terms get high weight.
- **Cosine Similarity** measures the angle between two vectors, making it length-independent and ideal for text comparison.

---

## 📁 Dataset

The `data/faqs.json` file contains 20 FAQ pairs across 6 categories:

| Category | Count |
|---|---|
| Account | 4 |
| Billing | 4 |
| Technical | 4 |
| Privacy | 2 |
| Features | 3 |
| Support | 3 |

**To add your own FAQs**, edit `data/faqs.json` following this format:
```json
{
  "id": 21,
  "category": "YourCategory",
  "question": "Your question here?",
  "answer": "Your detailed answer here."
}
```

---

## 📊 Query Logging

All queries are logged to `logs/query_log.csv`:

| Column | Description |
|---|---|
| timestamp | Date and time of query |
| user_query | Raw user input |
| matched_question | Best-matched FAQ question |
| confidence | Similarity score (0–100) |
| category | FAQ category |
| is_fallback | Whether fallback was used |

Use this data to identify gaps in your FAQ dataset.

---

## 🔧 Configuration

| Parameter | Default | Description |
|---|---|---|
| `confidence_threshold` | `0.2` | Min score to return a match |
| `ngram_range` | `(1, 2)` | Unigrams + bigrams in TF-IDF |
| `max_features` | `5000` | Max vocabulary size |
| `use_lemmatization` | `True` | Apply WordNet lemmatization |

---

## 🚀 Upgrade Roadmap

| Level | Upgrade | Impact |
|---|---|---|
| 🟡 Intermediate | Replace TF-IDF with **BM25** | Better keyword ranking |
| 🟠 Advanced | **Sentence-BERT** embeddings | True semantic understanding |
| 🔴 Expert | Fine-tune **DistilBERT** on your FAQ data | Near-human accuracy |
| 🔴 Expert | Add **intent classification** layer | Multi-intent support |
| 🟡 Intermediate | **Redis** caching for frequent queries | Performance at scale |
| 🟡 Intermediate | **Spell correction** (pyspellchecker) | Handles typos |
| 🟠 Advanced | **Multi-language** support (mBART) | Global user base |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — feel free to use this project for personal, academic, or commercial purposes.

---

## 👤 Author

Built as an internship project demonstrating practical NLP engineering skills.

> ⭐ If you found this helpful, please give it a star!
