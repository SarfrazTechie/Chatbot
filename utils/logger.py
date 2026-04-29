"""
utils/logger.py
---------------
Logs all user queries, matched answers, confidence scores,
and timestamps to a CSV file for analysis.

Use cases:
- Track unanswered / low-confidence queries
- Improve FAQ dataset over time
- Monitor chatbot performance

Author: FAQ Chatbot Project
"""

import csv
import os
import logging
from datetime import datetime


# Configure Python's standard logging for console/debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


class QueryLogger:
    """
    Logs user queries and chatbot responses to a CSV file.
    
    CSV columns:
    - timestamp: When the query was made
    - user_query: The raw user input
    - matched_question: The best-matched FAQ question
    - confidence: Similarity score (0–100)
    - category: FAQ category of the match
    - is_fallback: Whether a fallback response was used
    """

    def __init__(self, log_dir: str = "logs", log_file: str = "query_log.csv"):
        """
        Initialize the logger.
        
        Args:
            log_dir (str): Directory to store log files.
            log_file (str): Name of the CSV log file.
        """
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_file)
        self._ensure_log_file()

    def _ensure_log_file(self):
        """
        Create the log directory and CSV file with headers if they don't exist.
        """
        os.makedirs(self.log_dir, exist_ok=True)

        # Only write headers if the file is new/empty
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'user_query',
                    'matched_question',
                    'confidence',
                    'category',
                    'is_fallback'
                ])
            logger.info(f"Log file created at: {self.log_path}")

    def log(self, user_query: str, response: dict):
        """
        Log a single query-response pair.
        
        Args:
            user_query (str): The raw user question.
            response (dict): The response dict returned by FAQMatcher.get_answer().
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        matched_q = response.get('matched_question') or 'N/A'
        confidence = response.get('confidence', 0.0)
        category = response.get('category', 'Unknown')
        is_fallback = response.get('is_fallback', False)

        # Write to CSV
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                user_query,
                matched_q,
                confidence,
                category,
                is_fallback
            ])

        # Also log to console
        if is_fallback:
            logger.warning(f"FALLBACK | Query: '{user_query}'")
        else:
            logger.info(f"MATCH ({confidence}%) | Query: '{user_query}' → '{matched_q}'")

    def get_stats(self) -> dict:
        """
        Read the log file and compute basic statistics.
        
        Returns:
            dict: Stats including total queries, fallback rate, avg confidence.
        """
        if not os.path.exists(self.log_path):
            return {}

        rows = []
        with open(self.log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {'total_queries': 0}

        total = len(rows)
        fallbacks = sum(1 for r in rows if r['is_fallback'] == 'True')
        confidences = [
            float(r['confidence']) for r in rows
            if r['confidence'] and r['is_fallback'] == 'False'
        ]

        return {
            'total_queries': total,
            'fallback_count': fallbacks,
            'fallback_rate': round((fallbacks / total) * 100, 1),
            'avg_confidence': round(sum(confidences) / len(confidences), 1) if confidences else 0
        }

    def get_recent(self, n: int = 10) -> list:
        rows = self._read_rows()
        recent = rows[-n:][::-1]
        return [
            {
                'query':      r['user_query'],
                'confidence': r.get('confidence', '0'),
                'is_fallback': r.get('is_fallback') == 'True',
                'timestamp':  r.get('timestamp', ''),
            }
            for r in recent
        ]

    def _read_rows(self) -> list:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
