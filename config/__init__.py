"""
Configuration Module for Restaurant RAG System
"""
from .settings import (
    # Qdrant
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    USE_QDRANT_CLOUD,
    
    # Paths
    ENHANCED_FEATURES_PATH,
    
    # Timezone
    NZ_TZ,
    get_nz_now,
    
    # Models
    CROSS_ENCODER_MODEL,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    
    # Search Parameters
    BM25_TOP_K,
    VECTOR_TOP_K,
    RRF_K,
    MAX_CACHE_SIZE,
    CACHE_TTL_SECONDS,
    CACHE_SIMILARITY_THRESHOLD,
    MAX_RETRIEVE_K,
    MIN_RETRIEVE_K,
    REVIEW_BUFFER,
    DIVERSITY_MAX_PER_RESTAURANT,
    CROSS_ENCODER_TOP_K,
    EMBEDDING_CACHE_SIZE,
)

from .logging_config import (
    setup_logging,
    get_logger,
    setup_streamlit_logging,
)

__all__ = [
    # Qdrant
    'COLLECTION_NAME',
    'EMBEDDING_MODEL',
    'QDRANT_HOST',
    'QDRANT_PORT',
    'QDRANT_URL',
    'QDRANT_API_KEY',
    'USE_QDRANT_CLOUD',
    
    # Paths
    'ENHANCED_FEATURES_PATH',
    
    # Timezone
    'NZ_TZ',
    'get_nz_now',
    
    # Models
    'CROSS_ENCODER_MODEL',
    'OPENAI_CHAT_MODEL',
    'OPENAI_EMBEDDING_MODEL',
    
    # Search Parameters
    'BM25_TOP_K',
    'VECTOR_TOP_K',
    'RRF_K',
    'MAX_CACHE_SIZE',
    'CACHE_TTL_SECONDS',
    'CACHE_SIMILARITY_THRESHOLD',
    'MAX_RETRIEVE_K',
    'MIN_RETRIEVE_K',
    'REVIEW_BUFFER',
    'DIVERSITY_MAX_PER_RESTAURANT',
    'CROSS_ENCODER_TOP_K',
    'EMBEDDING_CACHE_SIZE',
    
    # Logging
    'setup_logging',
    'get_logger',
    'setup_streamlit_logging',
]