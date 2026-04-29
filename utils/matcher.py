"""
utils/matcher.py  —  UPGRADED v2.0
- Returns TOP 3 matches (not just 1)
- Adds "Did you mean?" suggestions on low confidence
- get_answer() response dict now includes top_matches + did_you_mean
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocessor import TextPreprocessor

DID_YOU_MEAN_THRESHOLD = 0.12   # weak signal band: show suggestions but no answer


class FAQMatcher:
    """TF-IDF + Cosine Similarity FAQ matcher. Returns top-3 results."""

    def __init__(self, faq_path: str, confidence_threshold: float = 0.2):
        self.confidence_threshold = confidence_threshold
        self.preprocessor = TextPreprocessor(use_lemmatization=True)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=5000, sublinear_tf=True
        )
        self.faqs = self._load_faqs(faq_path)
        self.processed_questions = self._preprocess_questions()
        self.tfidf_matrix = self._build_tfidf_matrix()

    def _load_faqs(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        print(f"[FAQMatcher] Loaded {len(faqs)} FAQs from '{path}'")
        return faqs

    def _preprocess_questions(self):
        return [self.preprocessor.preprocess(faq['question']) for faq in self.faqs]

    def _build_tfidf_matrix(self):
        matrix = self.vectorizer.fit_transform(self.processed_questions)
        print(f"[FAQMatcher] TF-IDF matrix built: {matrix.shape}")
        return matrix

    def get_answer(self, user_query: str) -> dict:
        """
        Returns best match + top 3 alternatives + "Did you mean?" suggestions.

        Response keys:
            answer           - best answer text (or fallback)
            matched_question - str or None
            confidence       - float 0-100
            category         - str
            is_fallback      - bool
            top_matches      - list[dict]  always present, max 3 items
            did_you_mean     - list[str]   populated on low-confidence queries
        """
        processed_query = self.preprocessor.preprocess(user_query)
        if not processed_query.strip():
            return self._fallback_response("empty query")

        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        # Always build top-3 for the UI
        top_matches = self._compute_top_matches(similarities, top_n=3)

        if best_score < self.confidence_threshold:
            # Weak signal — surface questions as "Did you mean?" hints
            did_you_mean = (
                [m['question'] for m in top_matches if m['confidence'] > 0]
                if best_score >= DID_YOU_MEAN_THRESHOLD else []
            )
            fb = self._fallback_response(f"low confidence: {best_score:.3f}")
            fb['did_you_mean'] = did_you_mean
            fb['top_matches'] = top_matches
            return fb

        best_faq = self.faqs[best_idx]
        return {
            'answer': best_faq['answer'],
            'matched_question': best_faq['question'],
            'confidence': round(best_score * 100, 2),
            'category': best_faq.get('category', 'General'),
            'is_fallback': False,
            'top_matches': top_matches,
            'did_you_mean': []
        }

    def _compute_top_matches(self, similarities: np.ndarray, top_n: int = 3) -> list:
        top_indices = np.argsort(similarities)[::-1][:top_n]
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:
                faq = self.faqs[idx]
                results.append({
                    'question': faq['question'],
                    'answer': faq['answer'],
                    'confidence': round(score * 100, 2),
                    'category': faq.get('category', 'General')
                })
        return results

    # Kept for backward compatibility
    def get_top_matches(self, user_query: str, top_n: int = 3) -> list:
        processed_query = self.preprocessor.preprocess(user_query)
        if not processed_query.strip():
            return []
        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        return self._compute_top_matches(similarities, top_n)

    def _fallback_response(self, reason: str = "") -> dict:
        return {
            'answer': (
                "I couldn't find a relevant answer to your question. "
                "Please try rephrasing, or contact support@company.com."
            ),
            'matched_question': None,
            'confidence': 0.0,
            'category': 'Fallback',
            'is_fallback': True,
            'top_matches': [],
            'did_you_mean': [],
            '_reason': reason
        }
