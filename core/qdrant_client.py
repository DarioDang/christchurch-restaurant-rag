"""
Qdrant Client Initialization and Index Management
Handles connection, document loading, and BM25 index building
"""

import pandas as pd 
import string 
from typing import Dict, List, Tuple
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# # Import from config 
# from config import (
#     COLLECTION_NAME,
#     EMBEDDING_MODEL,
#     QDRANT_HOST,
#     QDRANT_PORT,
#     ENHANCED_FEATURES_PATH,
# )

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    EMBEDDING_MODEL,
    USE_QDRANT_CLOUD,
    COLLECTION_NAME,
    ENHANCED_FEATURES_PATH
)


def get_stopwords():
    """Get English stopwords (basic set)"""
    return {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'at', 'for', 'to', 'of', 
        'on', 'with', 'as', 'by', 'is', 'was', 'are', 'were', 'be', 'been',
        'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }


def _build_bm25_corpus(all_docs: List[Dict]) -> Tuple[List[List[str]], List[str]]:
    """
    Build BM25 corpus with improved tokenization
    
    Args:
        all_docs: List of document dictionaries with 'id' and 'payload'
    
    Returns:
        Tuple of (corpus, doc_ids)
    """
    stop_words = get_stopwords()
    corpus = []
    doc_ids = []
    
    for doc in all_docs:
        text = doc['payload'].get('text', '')
        # Remove punctuation and tokenize
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = [
            word.lower() for word in text.split() 
            if word.lower() not in stop_words and len(word) > 2
        ]
        corpus.append(tokens)
        doc_ids.append(doc['id'])
    
    return corpus, doc_ids


def init_qdrant():
    """
    Initialize Qdrant client and load metadata.
    Also loads Tier 1 enhanced features DataFrame.
    
    Returns:
        Tuple of (client, model, bm25_index, doc_ids, doc_lookup, metadata)
    """
    # Initialize Qdrant client (supports both cloud and local)
    if USE_QDRANT_CLOUD:
        # Use Qdrant Cloud
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30
        )
        print(f"Connected to Qdrant Cloud")
    else:
        # Use local Qdrant
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )
        print(f"Connected to local Qdrant")

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Verify collection exists 
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Connected to Qdrant: {collection_info.points_count} documents")
    except Exception as e:
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' not found. "
            f"Run 02_build_qdrant_index.ipynb first. Error: {e}"
        )
    
    # Load all documents for BM25 
    print("Loading documents for BM25...")
    all_docs = []
    offset = None 

    while True:
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records, next_offset = results 

        for record in records:
            all_docs.append({
                'id': record.id,
                'payload': record.payload
            })
        if next_offset is None:
            break
        
        offset = next_offset
    
    # Build BM25 index with improved tokenization
    print("Building BM25 index with enhanced tokenization...")
    corpus, doc_ids = _build_bm25_corpus(all_docs)
    bm25_index = BM25Okapi(corpus)
    doc_lookup = {doc['id']: doc for doc in all_docs}

    # Extract metadata
    all_restaurants = list(set(
        doc['payload']['restaurant'] 
        for doc in all_docs
    ))

    all_cuisines = set()
    for doc in all_docs:
        all_cuisines.update(doc['payload'].get('cuisines', []))
    
    metadata = {
        'all_restaurants': sorted(all_restaurants),
        'all_cuisines': sorted(list(all_cuisines)),
        'total_docs': len(all_docs)
    }
    
    print(f"✓ Found {len(all_restaurants)} restaurants")
    print(f"✓ Found {len(all_cuisines)} cuisines")
    
    # Load Tier 1 enhanced features
    print("\nLoading Tier 1 enhanced features...")
    try:
        enhanced_df = pd.read_parquet(ENHANCED_FEATURES_PATH)
        
        print(f"✓ Loaded {len(enhanced_df)} restaurants with pre-processed Tier 1 features")
        
        # Check which Tier 1 columns are present
        tier1_cols = [
            'restaurant', 'restaurant_id', 'address', 'cuisines', 'lat', 'lon',
            'phone', 'types_normalized', 'price_level_numeric', 'price_bucket',
            'hours_category', 'open_state',
            'hours_dict', 'hours_pretty',                      
            'has_delivery', 'has_takeout', 'has_dine_in',
            'avg_rating', 'num_reviews'                        
        ]

        present_cols = [col for col in tier1_cols if col in enhanced_df.columns]
        print(f"✓ Tier 1 features available: {len(present_cols)}/{len(tier1_cols)}")
        
        metadata['enhanced_df'] = enhanced_df
        metadata['tier1_enabled'] = len(present_cols) > 0
        
    except Exception as e:
        print(f"Could not load Tier 1 features: {e}")
        print("Continuing without Tier 1 filtering...")
        metadata['enhanced_df'] = None
        metadata['tier1_enabled'] = False
    
    return client, model, bm25_index, doc_ids, doc_lookup, metadata