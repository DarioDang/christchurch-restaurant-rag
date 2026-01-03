"""
Query Analytics & Trending - Qdrant Cloud Version
Stores analytics in Qdrant instead of SQLite
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, PayloadSchemaType
import uuid
import time


class QueryAnalytics:
    """
    Query tracking using Qdrant Cloud
    Stores queries as points in a separate collection
    """

    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        """
        Initialize analytics with Qdrant
        
        Args:
            qdrant_client: Existing Qdrant client (reuse from main app)
        """
        self.collection_name = "query_analytics"

        if qdrant_client:
            self.client = qdrant_client
        else:
            # Create new client from env vars
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if qdrant_url and qdrant_api_key:
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                print("⚠️ Qdrant credentials not found - analytics disabled")
                self.client = None
                return
        
        self._init_collection()
    
    def _init_collection(self):
        """Create analytics collection if it doesn't exist"""
        if not self.client:
            return
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                # Create collection (dummy vector since we only use payload)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE)
                )
                print(f"✅ Created analytics collection: {self.collection_name}")
                
                # ✅ FIX: Create index on timestamp field for Range filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="timestamp",
                    field_schema=PayloadSchemaType.FLOAT
                )
                print(f"✅ Created index on 'timestamp' field")
            else:
                print(f"✅ Analytics collection ready: {self.collection_name}")
                
                # ✅ Ensure index exists even if collection was created before
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="timestamp",
                        field_schema=PayloadSchemaType.FLOAT
                    )
                    print(f"✅ Ensured index on 'timestamp' field")
                except Exception as idx_error:
                    # Index might already exist, that's okay
                    if "already exists" not in str(idx_error).lower():
                        print(f"⚠️ Index creation info: {idx_error}")
                
        except Exception as e:
            print(f"⚠️ Failed to initialize analytics collection: {e}")
    
    def normalize_query(self, query: str) -> str:
        """Normalize query for grouping"""
        q = query.lower().strip()
        
        # Remove location keywords
        location_words = [
            'nearby', 'near me', 'close by', 'around here',
            'in the area', 'close to me', 'around me',
            'walking distance', 'driving distance', 'close', 'near', 'around'
        ]
        
        for word in location_words:
            q = q.replace(word, '')
        
        # Remove punctuation
        q = re.sub(r'[^\w\s]', '', q)
        
        # Remove extra spaces
        q = ' '.join(q.split())
        
        return q.strip()
    
    def log_query(self, query: str, session_id: Optional[str] = None,
                  location_enabled: bool = False, results_count: int = 0):
        """Log a user query to Qdrant"""
        if not self.client:
            return
        
        normalized = self.normalize_query(query)

        # Skip if normalized query is too short
        if len(normalized) < 3 or normalized in ['best', 'good', 'find', 'show']:
            return
        
        try:
            # Use Unix timestamp (float)
            timestamp = time.time()
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0],  # Dummy vector (we only use payload)
                payload={
                    'query_text': query,
                    'normalized_query': normalized,
                    'timestamp': timestamp,
                    'session_id': session_id,
                    'location_enabled': location_enabled,
                    'results_count': results_count
                }
            )

            # Insert into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            print(f"📊 Logged query to Qdrant: '{normalized}'")
            
        except Exception as e:
            print(f"⚠️ Failed to log query: {e}")
    
    def get_trending_queries(self, 
                            time_window_hours: int = 24,
                            min_count: int = 2,
                            top_n: int = 3) -> List[Dict]:
        """Get trending queries from Qdrant"""
        if not self.client:
            return []
        
        try:
            # Calculate cutoff as Unix timestamp (float)
            cutoff = time.time() - (time_window_hours * 3600)

            # Scroll all recent queries
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="timestamp",
                            range=Range(gte=cutoff)
                        )
                    ]
                ),
                limit=1000,  # Get up to 1000 recent queries
                with_payload=True,
                with_vectors=False
            )

            # Count normalized queries
            query_counts = Counter()
            query_examples = {}
            
            for point in results:
                normalized = point.payload.get('normalized_query', '')
                query_text = point.payload.get('query_text', '')
                
                query_counts[normalized] += 1
                query_examples[normalized] = query_text  # Keep one example
            
            # Get top queries
            trending = []
            for normalized, count in query_counts.most_common(top_n):
                if count >= min_count:
                    trending.append({
                        'normalized_query': normalized,
                        'count': count,
                        'example_query': query_examples[normalized],
                        'display_text': self._format_display_text(query_examples[normalized])
                    })
            
            return trending
            
        except Exception as e:
            print(f"⚠️ Failed to get trending queries: {e}")
            return []
    
    def _format_display_text(self, query: str) -> str:
        """Format query for display"""
        query = query.strip()
        if query:
            query = query[0].upper() + query[1:]
        
        if len(query) > 40:
            query = query[:37] + "..."
        
        return query
    
    def get_fallback_queries(self) -> List[Dict]:
        """Return fallback queries when not enough data"""
        return [
            {
                'example_query': 'Best pizza nearby',
                'display_text': 'Best pizza nearby',
                'count': 0
            },
            {
                'example_query': 'Sushi restaurants',
                'display_text': 'Sushi restaurants',
                'count': 0
            },
            {
                'example_query': 'Korean BBQ',
                'display_text': 'Korean BBQ',
                'count': 0
            }
        ]
    
    def get_stats(self) -> Dict:
        """Get analytics statistics"""
        if not self.client:
            return {'total_queries': 0, 'queries_24h': 0, 'unique_queries': 0}
        
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            total = collection_info.points_count
            
            # Count queries in last 24 hours using Unix timestamp
            cutoff = time.time() - (24 * 3600)
            
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="timestamp",
                            range=Range(gte=cutoff)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            last_24h = len(results)
            
            # Count unique queries
            unique = len(set(point.payload.get('normalized_query', '') for point in results))
            
            return {
                'total_queries': total,
                'queries_24h': last_24h,
                'unique_queries': unique
            }
            
        except Exception as e:
            print(f"⚠️ Failed to get stats: {e}")
            return {'total_queries': 0, 'queries_24h': 0, 'unique_queries': 0}


# Global instance
_analytics = None

def get_analytics(qdrant_client: Optional[QdrantClient] = None) -> QueryAnalytics:
    """Get or create analytics instance"""
    global _analytics
    if _analytics is None:
        _analytics = QueryAnalytics(qdrant_client=qdrant_client)
    return _analytics