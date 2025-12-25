"""
Configuration Settings for Restaurant RAG System
All constants extracted from tools.py
"""

import os 
from dotenv import load_dotenv
from pathlib import Path
import pytz
from datetime import datetime, timezone 

# ============================================
# LOAD .ENV FROM PROJECT ROOT
# ============================================

# Get project root (parent of config directory)
config_dir = Path(__file__).parent  # /path/to/config/
project_root = config_dir.parent    # /path/to/restaurants-rag-production/

# Load .env from project root
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Verify .env was loaded
if not env_path.exists():
    print(f"WARNING: .env file not found at {env_path}")
    print("   Using environment variables or defaults")
else:
    print(f"Loaded .env from: {env_path}")

# ============================================
# QDRANT CONFIGURATION
# ============================================
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "christchurch_restaurants")

# Qdrant Cloud (recommended for production)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Fallback to local Qdrant (for development)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Determine which mode we're in
USE_QDRANT_CLOUD = bool(QDRANT_URL and QDRANT_API_KEY)

if USE_QDRANT_CLOUD:
    print(f"✅ Using Qdrant Cloud: {QDRANT_URL}")
else:
    print(f"✅ Using Local Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")

# ============================================
# PATHS
# ============================================
# Auto-detect if running in Docker or locally
if os.path.exists("/app/data"):
    ENHANCED_FEATURES_PATH = "/app/data/chc_restaurants_enriched_features.parquet"
else:
    ENHANCED_FEATURES_PATH = "data/chc_restaurants_enriched_features.parquet"


# ============================================
# TIMEZONE CONFIGURATION
# ============================================
NZ_TZ = pytz.timezone('Pacific/Auckland')

def get_nz_now():
    """Get current time in NZ timezone - works in Docker too"""
    utc_now = datetime.now(timezone.utc)
    nz_now = utc_now.astimezone(NZ_TZ)
    return nz_now


# ============================================
# MODEL CONFIGURATION
# ============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" 
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"


# ============================================
# SEARCH PARAMETERS
# ============================================
# BM25 Configuration
BM25_TOP_K = 20

# Vector Search Configuration
VECTOR_TOP_K = 20

# Reciprocal Rank Fusion
RRF_K = 60

# Cache Configuration
MAX_CACHE_SIZE = 100
CACHE_TTL_SECONDS = 3600  # 1 hour
CACHE_SIMILARITY_THRESHOLD = 0.92

# Search Parameters
MAX_RETRIEVE_K = 15
MIN_RETRIEVE_K = 5
REVIEW_BUFFER = 3

# Result Diversity
DIVERSITY_MAX_PER_RESTAURANT = 2

# Cross-encoder Reranking
CROSS_ENCODER_TOP_K = 5

# Embedding Cache
EMBEDDING_CACHE_SIZE = 200

