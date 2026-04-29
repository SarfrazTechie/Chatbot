"""
tests/test_chatbot.py
---------------------
Unit tests for FAQ Chatbot components.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import json
import tempfile
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessor import TextPreprocessor
from utils.matcher import FAQMatcher
from utils.logger import QueryLogger


# ─── TextPreprocessor Tests ──────────────────────────────────────────────────

class TestTextPreprocessor:

    def setup_method(self):
        self.preprocessor = TextPreprocessor(use_lemmatization=True)

    def test_clean_text_lowercases(self):
        result = self.preprocessor.clean_text("Hello WORLD")
        assert result == "hello world"

    def test_clean_text_removes_special_chars(self):
        result = self.preprocessor.clean_text("How do I reset my password??!")
        assert "?" not in result
        assert "!" not in result

    def test_clean_text_removes_digits(self):
        result = self.preprocessor.clean_text("Call us at 12345")
        assert "12345" not in result

    def test_tokenize_returns_list(self):
        tokens = self.preprocessor.tokenize("hello world")
        assert isinstance(tokens, list)
        assert "hello" in tokens

    def test_remove_stopwords(self):
        tokens = ["how", "do", "i", "reset", "password"]
        filtered = self.preprocessor.remove_stopwords(tokens)
        # "how", "do", "i" are stopwords
        assert "reset" in filtered
        assert "password" in filtered
        assert "i" not in filtered

    def test_lemmatization(self):
        tokens = ["running", "payments", "better"]
        lemmatized = self.preprocessor.apply_lemmatization(tokens)
        assert isinstance(lemmatized, list)

    def test_full_pipeline_returns_string(self):
        result = self.preprocessor.preprocess("How do I reset my password?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_input(self):
        result = self.preprocessor.preprocess("")
        assert result == ""


# ─── FAQMatcher Tests ────────────────────────────────────────────────────────

@pytest.fixture
def sample_faq_file():
    """Create a temporary FAQ JSON file for testing."""
    faqs = [
        {
            "id": 1,
            "category": "Account",
            "question": "How do I reset my password?",
            "answer": "Click 'Forgot Password' on the login page."
        },
        {
            "id": 2,
            "category": "Billing",
            "question": "What payment methods do you accept?",
            "answer": "We accept Visa, MasterCard, and PayPal."
        },
        {
            "id": 3,
            "category": "Support",
            "question": "How do I contact support?",
            "answer": "Email us at support@company.com."
        }
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(faqs, f)
        return f.name


class TestFAQMatcher:

    def setup_method(self, sample_faq_file=None):
        """Set up a fresh matcher before each test."""
        # We'll use the fixture via a parameter approach
        pass

    def test_load_faqs(self, sample_faq_file):
        matcher = FAQMatcher(sample_faq_file, confidence_threshold=0.1)
        assert len(matcher.faqs) == 3

    def test_returns_dict(self, sample_faq_file):
        matcher = FAQMatcher(sample_faq_file, confidence_threshold=0.1)
        result = matcher.get_answer("How do I reset my password?")
        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'confidence' in result
        assert 'is_fallback' in result

    def test_good_match(self, sample_faq_file):
        matcher = FAQMatcher(sample_faq_file, confidence_threshold=0.1)
        result = matcher.get_answer("reset password")
        assert result['is_fallback'] == False
        assert result['confidence'] > 0

    def test_fallback_on_unrelated_query(self, sample_faq_file):
        matcher = FAQMatcher(sample_faq_file, confidence_threshold=0.9)  # Very high threshold
        result = matcher.get_answer("xyzzy foo bar nonsense")
        assert result['is_fallback'] == True
        assert result['confidence'] == 0.0

    def test_fallback_on_empty_query(self, sample_faq_file):
        matcher = FAQMatcher(sample_faq_file)
        result = matcher.get_answer("")
        assert result['is_fallback'] == True

    def test_top_matches_returns_list(self, sample_faq_file):
        matcher = FAQMatcher(sample_faq_file, confidence_threshold=0.1)
        results = matcher.get_top_matches("password", top_n=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_confidence_is_percentage(self, sample_faq_file):
        matcher = FAQMatcher(sample_faq_file, confidence_threshold=0.1)
        result = matcher.get_answer("reset password")
        if not result['is_fallback']:
            assert 0 <= result['confidence'] <= 100


# ─── QueryLogger Tests ───────────────────────────────────────────────────────

class TestQueryLogger:

    def setup_method(self):
        """Create a temp directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = QueryLogger(log_dir=self.temp_dir, log_file="test_log.csv")

    def test_log_creates_file(self):
        log_path = os.path.join(self.temp_dir, "test_log.csv")
        assert os.path.exists(log_path)

    def test_log_writes_row(self):
        response = {
            'answer': 'Test answer',
            'matched_question': 'Test question',
            'confidence': 85.5,
            'category': 'Test',
            'is_fallback': False
        }
        self.logger.log("test query", response)
        stats = self.logger.get_stats()
        assert stats['total_queries'] == 1

    def test_stats_fallback_rate(self):
        fallback_response = {
            'answer': 'Fallback', 'matched_question': None,
            'confidence': 0.0, 'category': 'Fallback', 'is_fallback': True
        }
        good_response = {
            'answer': 'Good answer', 'matched_question': 'Test Q',
            'confidence': 75.0, 'category': 'General', 'is_fallback': False
        }
        self.logger.log("query 1", good_response)
        self.logger.log("query 2", fallback_response)

        stats = self.logger.get_stats()
        assert stats['total_queries'] == 2
        assert stats['fallback_count'] == 1
        assert stats['fallback_rate'] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
