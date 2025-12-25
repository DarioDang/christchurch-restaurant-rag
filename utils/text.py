"""
Text Processing Utilities
Handles text cleaning, tokenization, and normalization
"""

import re
import emoji
import string

def shorten(text, max_length=50):
    """
    Shorten text for logging
    
    Args:
        text: Text to shorten
        max_length: Maximum length
    
    Returns:
        Shortened text with ellipsis if needed
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def clean_text(text):
    """
    Remove emoji, newlines, and excessive whitespace
    
    Args:
        text: Text to clean
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_stopwords():
    """
    Get English stopwords (basic set)
    
    Returns:
        Set of stopwords
    """
    return {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'at', 'for', 'to', 'of', 
        'on', 'with', 'as', 'by', 'is', 'was', 'are', 'were', 'be', 'been',
        'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }


def tokenize_text(text, remove_stopwords=True, min_length=2):
    """
    Tokenize text for BM25 search
    
    Args:
        text: Text to tokenize
        remove_stopwords: Whether to remove stopwords
        min_length: Minimum token length
    
    Returns:
        List of tokens
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = [word.lower() for word in text.split()]
    
    # Filter stopwords and short tokens
    if remove_stopwords:
        stop_words = get_stopwords()
        tokens = [
            word for word in tokens 
            if word not in stop_words and len(word) > min_length
        ]
    else:
        tokens = [word for word in tokens if len(word) > min_length]
    
    return tokens


def normalize_query(query):
    """
    Normalize query for cache key generation
    
    Args:
        query: Search query
    
    Returns:
        Normalized query string
    """
    # Lowercase and strip
    normalized = query.lower().strip()
    
    # Remove punctuation
    normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
    
    # Remove extra spaces
    normalized = ' '.join(normalized.split())
    
    return normalized