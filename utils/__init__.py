"""
Utilities Module - Helper Functions
"""
from .text import (
    shorten,
    clean_text,
    get_stopwords,
    tokenize_text,
    normalize_query,
)

from .geo import (
    haversine_distance,
    get_distance_category,
    detect_location_intent,
    calculate_bounding_box,
)

from .hours import (
    parse_hours_string,
    is_restaurant_open_now,
    compute_temporal_context,
    calculate_is_open_now,
)

from .caching import QueryCache

__all__ = [
    # Text utilities
    'shorten',
    'clean_text',
    'get_stopwords',
    'tokenize_text',
    'normalize_query',
    
    # Geo utilities
    'haversine_distance',
    'get_distance_category',
    'detect_location_intent',
    'calculate_bounding_box',
    
    # Hours utilities
    'parse_hours_string',
    'is_restaurant_open_now',
    'compute_temporal_context',
    'calculate_is_open_now',
    
    # Caching
    'QueryCache',
]