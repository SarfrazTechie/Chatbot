"""
utils/preprocessor.py
---------------------
Handles all NLP text preprocessing:
- Tokenization
- Lowercasing
- Stopword removal
- Stemming and Lemmatization

Author: FAQ Chatbot Project
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources (runs once)
def download_nltk_resources():
    """Download all required NLTK data packages."""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

download_nltk_resources()


class TextPreprocessor:
    """
    A reusable text preprocessing pipeline for NLP tasks.
    
    Features:
    - Cleans and normalizes raw text
    - Removes stopwords
    - Applies stemming or lemmatization
    """

    def __init__(self, use_stemming: bool = False, use_lemmatization: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            use_stemming (bool): Apply Porter Stemmer if True.
            use_lemmatization (bool): Apply WordNet Lemmatizer if True.
                                      Lemmatization is preferred over stemming
                                      for better word quality.
        """
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

    def clean_text(self, text: str) -> str:
        """
        Remove special characters, digits, and extra whitespace.
        
        Args:
            text (str): Raw input text.
        
        Returns:
            str: Cleaned text.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> list:
        """
        Tokenize text into individual words.
        
        Args:
            text (str): Cleaned text string.
        
        Returns:
            list: List of word tokens.
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list) -> list:
        """
        Remove common English stopwords from token list.
        
        Args:
            tokens (list): List of word tokens.
        
        Returns:
            list: Filtered list without stopwords.
        """
        return [token for token in tokens if token not in self.stop_words]

    def apply_stemming(self, tokens: list) -> list:
        """
        Reduce words to their root form using Porter Stemmer.
        Example: 'running' -> 'run', 'payments' -> 'payment'
        
        Args:
            tokens (list): List of word tokens.
        
        Returns:
            list: Stemmed tokens.
        """
        return [self.stemmer.stem(token) for token in tokens]

    def apply_lemmatization(self, tokens: list) -> list:
        """
        Reduce words to their base dictionary form using WordNet.
        Example: 'better' -> 'good', 'running' -> 'run'
        
        Args:
            tokens (list): List of word tokens.
        
        Returns:
            list: Lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline: clean → tokenize → 
        remove stopwords → stem/lemmatize → rejoin.
        
        Args:
            text (str): Raw input text.
        
        Returns:
            str: Fully preprocessed text as a single string.
        """
        # Step 1: Clean the text
        cleaned = self.clean_text(text)

        # Step 2: Tokenize
        tokens = self.tokenize(cleaned)

        # Step 3: Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Step 4: Apply stemming or lemmatization
        if self.use_stemming and self.stemmer:
            tokens = self.apply_stemming(tokens)
        elif self.use_lemmatization and self.lemmatizer:
            tokens = self.apply_lemmatization(tokens)

        # Step 5: Rejoin tokens into a single string (required for TF-IDF)
        return ' '.join(tokens)
