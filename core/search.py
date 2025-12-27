"""
RestaurantSearchTools Class
Main search engine with hybrid BM25+Vector search, Tier 1 filtering, and location awareness
"""

# Just add these imports at the top:
import json
import re
import math
import time
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from tracing import tracer
from functools import lru_cache
from sentence_transformers import CrossEncoder
from datetime import datetime
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

# Import from config
from config import (
    COLLECTION_NAME,
    get_nz_now,
    CROSS_ENCODER_MODEL,
    BM25_TOP_K,
    VECTOR_TOP_K,
    RRF_K,
    MAX_CACHE_SIZE,
    CACHE_TTL_SECONDS,
    CROSS_ENCODER_TOP_K,
    DIVERSITY_MAX_PER_RESTAURANT,
    MAX_RETRIEVE_K,
    MIN_RETRIEVE_K,
    REVIEW_BUFFER,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL,
)

from utils import (
    get_stopwords,
    haversine_distance,
    get_distance_category,
    detect_location_intent,
    parse_hours_string,
    is_restaurant_open_now,
    compute_temporal_context,
    calculate_is_open_now,
)



class RestaurantSearchTools:
    """
    Enhanced Qdrant-based hybrid search for restaurant reviews.
    Features:
    - Fixed restaurant name detection with cuisine validation
    - Improved BM25 tokenization
    - Query intent classification
    - Result diversity
    - Semantic query expansion
    - Fallback handling
    - Embedding caching
    """

    def __init__(self, qdrant_client, embedding_model, bm25_index, 
                 bm25_doc_ids, doc_lookup, metadata):
        self.client = qdrant_client
        self.model = embedding_model
        self.bm25 = bm25_index
        self.bm25_doc_ids = bm25_doc_ids
        self.doc_lookup = doc_lookup
        self.metadata = metadata
        self.all_restaurants = metadata['all_restaurants']
        self.all_cuisines = metadata['all_cuisines']
        self.openai_client = OpenAI()
        self.build_cuisine_map()
        self._build_restaurant_cuisine_index()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.query_cache = {}  # {query_embedding_tuple: (query, results, timestamp)}
        self.cache_threshold = 0.92  # High similarity needed
        
        # Tier 1: Enhanced features
        self.enhanced_df = metadata.get('enhanced_df')
        self.tier1_enabled = metadata.get('tier1_enabled', False)
        
        if self.tier1_enabled:
            # Create restaurant_id lookup for fast access
            self.restaurant_lookup = self.enhanced_df.set_index('restaurant_id').to_dict('index')
            print(f"✓ Tier 1 filtering enabled with {len(self.restaurant_lookup)} restaurants")
        else:
            self.restaurant_lookup = {}
        
        self._build_restaurant_bm25_index()


        # Add normalized restaurant lookup for O(1) existence checks
        self.normalized_restaurants = {
            self._normalize_restaurant_name(r): r 
            for r in self.all_restaurants
        }
        print(f"✓ Built normalized restaurant lookup for {len(self.normalized_restaurants)} restaurants")


    def _normalize_restaurant_name(self, name: str) -> str:
        """Normalize restaurant name for consistent comparison"""
        if not name:
            return ""
        return (name.lower()
                .replace("'", "").replace("'", "").replace("'", "")
                .strip())
    
    def _build_restaurant_bm25_index(self):
        """Build index: restaurant_name → [bm25_doc_indices]"""
        self.restaurant_bm25_index = {}
        
        for idx, doc_id in enumerate(self.bm25_doc_ids):
            doc = self.doc_lookup.get(doc_id)
            if doc:
                restaurant = doc['payload'].get('restaurant', '')
                restaurant_normalized = self._normalize_restaurant_name(restaurant)
                
                if restaurant_normalized not in self.restaurant_bm25_index:
                    self.restaurant_bm25_index[restaurant_normalized] = []
                
                self.restaurant_bm25_index[restaurant_normalized].append(idx)
        
        print(f"✓ Built BM25 restaurant index for {len(self.restaurant_bm25_index)} restaurants")

    # ========================================
    # TIER 1: FILTER DETECTION & APPLICATION
    # ========================================
    
    def detect_tier1_filter_intents(self, query: str) -> dict:
        """
        Detect Tier 1 filtering intents from user query.
        
        Returns dict with detected filters:
        {
            'hours_category': str,
            'delivery': bool,
            'takeout': bool,
            'dine_in': bool,
            'price_range': (min, max),
            'categories': list[str],
            'needs_hours_info': bool  # NEW: Signals hours query without filtering
        }
        """
        if not self.tier1_enabled:
            return {}
        
        query_lower = query.lower()
        intents = {}
        
        # General hours information requests
        # These don't filter but trigger enrichment for metadata availability
        hours_keywords = [
            'opening hour', 'opening time', 'open hour', 'open time',
            'hours of operation', 'operating hour', 'business hour',
            'what time', 'when open', 'when close', 'when does',
            'schedule', 'hours', 'timing',
            'weekend hour', 'weekday hour', 
            'saturday hour', 'sunday hour',
            'monday hour', 'friday hour'
        ]
        
        if any(keyword in query_lower for keyword in hours_keywords):
            # Mark that we need hours info (triggers enrichment even without filtering)
            intents['needs_hours_info'] = True
        
        if any(word in query_lower for word in ['open now', 'currently open', 'right now', 'is it open', 'are they open', 'open yet']):
            intents['check_open_now'] = True 

        if any(word in query_lower for word in ['late night', 'open late', 'after 10', 'after midnight', 'late hour']):
            intents['hours_category'] = 'late_night'
        
        if any(word in query_lower for word in ['breakfast', 'brunch', 'morning', 'early']):
            intents['hours_category'] = 'breakfast'
        
        # Service Options Detection
        if any(word in query_lower for word in ['delivery', 'deliver', 'delivered']):
            intents['delivery'] = True
        
        if any(word in query_lower for word in ['takeout', 'take out', 'takeaway', 'to go']):
            intents['takeout'] = True
        
        if any(word in query_lower for word in ['dine in', 'dine-in', 'sit down', 'eat in']):
            intents['dine_in'] = True
        
        # Price Range Detection
        if any(word in query_lower for word in ['cheap', 'budget', 'inexpensive', 'affordable']):
            intents['price_range'] = (1, 1)
        elif any(word in query_lower for word in ['expensive', 'upscale', 'fine dining', 'fancy', 'pricey']):
            intents['price_range'] = (3, 4)
        elif any(word in query_lower for word in ['mid-range', 'moderate', 'medium price']):
            intents['price_range'] = (2, 2)
        
        # Category Detection
        categories = []
        category_keywords = {
            'coffee': 'coffee_shop',
            'cafe': 'cafe',
            'bar': 'bar',
            'pub': 'pub',
            'italian': 'italian',
            'chinese': 'chinese',
            'japanese': 'japanese',
            'sushi': 'sushi',
            'pizza': 'pizza',
            'american': 'american',
            'seafood': 'seafood',
            'indian': 'indian',
            'thai': 'thai',
            'vietnamese': 'vietnamese',
            'korean': 'korean',
            'mexican': 'mexican',
            'vegan': 'vegan',
            'vegetarian': 'vegetarian',
        }
        
        for keyword, category in category_keywords.items():
            if keyword in query_lower:
                categories.append(category)
        
        if categories:
            intents['categories'] = categories
        
        return intents
    
    def apply_tier1_filters(self, intents: dict) -> Optional[Filter]:
        """
        Apply Tier 1 filters to Qdrant search based on detected intents.
        
        Returns Qdrant Filter object or None if no filters.
        """
        if not self.tier1_enabled or not intents:
            return None
        
        filtered_df = self.enhanced_df.copy()

        # NEW: Real-time "Is Open Now?" filtering
        if intents.get('check_open_now'):
            if 'hours_dict' in filtered_df.columns:
                # Calculate is_open_now in real-time for each restaurant
                filtered_df['is_open_now_realtime'] = filtered_df['hours_dict'].apply(
                    lambda hd: calculate_is_open_now(hd) if isinstance(hd, dict) else False
                )
                filtered_df = filtered_df[filtered_df['is_open_now_realtime'] == True]
                print(f"✓ Filtered to {len(filtered_df)} restaurants open RIGHT NOW")
        
        # Apply filters to DataFrame
        if intents.get('hours_category'):
            if 'hours_category' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['hours_category'] == intents['hours_category']]
        
        if intents.get('delivery'):
            if 'has_delivery' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['has_delivery'] == True]
        
        if intents.get('takeout'):
            if 'has_takeout' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['has_takeout'] == True]
        
        if intents.get('dine_in'):
            if 'has_dine_in' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['has_dine_in'] == True]
        
        if intents.get('price_range'):
            if 'price_level_numeric' in filtered_df.columns:
                min_price, max_price = intents['price_range']
                filtered_df = filtered_df[
                    (filtered_df['price_level_numeric'] >= min_price) &
                    (filtered_df['price_level_numeric'] <= max_price) &
                    (filtered_df['price_level_numeric'] > 0)
                ]
        
        if intents.get('categories'):
            if 'types_normalized' in filtered_df.columns:
                category_mask = filtered_df['types_normalized'].apply(
                    lambda types: any(cat in str(types).lower() for cat in intents['categories'])
                )
                filtered_df = filtered_df[category_mask]
        
        if len(filtered_df) == 0:
            return None
        
        # Build Qdrant filter using restaurant_ids
        restaurant_ids = filtered_df['restaurant_id'].unique().tolist()
        
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="restaurant_id",
                    match=MatchAny(any=restaurant_ids)
                )
            ]
        )
        
        return qdrant_filter
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate restaurants, keeping the highest scoring one.
        """
        seen = {}
        
        for result in results:
            restaurant_id = result.get('restaurant_id')
            
            if not restaurant_id:
                # If no restaurant_id, use restaurant name
                restaurant_id = result.get('restaurant', '').lower()
            
            if restaurant_id not in seen:
                seen[restaurant_id] = result
            else:
                # Keep the one with higher score
                if result.get('score', 0) > seen[restaurant_id].get('score', 0):
                    seen[restaurant_id] = result
        
        # Return as list, sorted by score
        deduplicated = list(seen.values())
        deduplicated.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return deduplicated
    
    def _format_metadata_text(self, tier1: dict, restaurant_name: str) -> str:
        """
        Format Tier 1 features as CONCISE, readable text for LLM.
        NOW INCLUDES: Real-time temporal context for open/closed status.
        Only includes essential information to avoid over-answering.
        
        UPDATED: Skip fields that are None/null to avoid bad data.
        """
        parts = [f"Restaurant: {restaurant_name}"]
        
        # Address (concise) - only if not None
        if tier1.get('address'):
            parts.append(f"Address: {tier1['address']}")

        # Phone (concise) - only if not None
        if tier1.get('phone'):
            parts.append(f"Phone: {tier1['phone']}")
        
        #  Only add hours if hours_dict is valid (not None, not all nulls)
        hours_dict = tier1.get('hours_dict')
        has_valid_hours = False
        
        if isinstance(hours_dict, dict) and hours_dict:
            # Check if at least one day has real hours
            has_valid_hours = any(
                v is not None and v != '' and str(v).lower() not in ['none', 'null', 'nan']
                for v in hours_dict.values()
            )
        
        if has_valid_hours:
            # Operating hours - use pre-formatted string
            if tier1.get('hours_pretty'):
                parts.append(f"Operating Hours: {tier1['hours_pretty']}")
            
            # Today's hours - use pre-parsed dict
            today_name = get_nz_now().strftime('%A').lower()
            today_hours = hours_dict.get(today_name)
            if today_hours and str(today_hours).lower() not in ['closed', 'unavailable', 'none', 'null', '']:
                today_name_formatted = get_nz_now().strftime('%A')
                parts.append(f"Today ({today_name_formatted}): {today_hours}")
        
        # Services (concise)
        services = []
        if tier1.get('has_dine_in'):
            services.append("Dine-in")
        if tier1.get('has_takeout'):
            services.append("Takeout")
        if tier1.get('has_delivery'):
            services.append("Delivery")
        if services:
            parts.append(f"Services: {', '.join(services)}")
        
        # Price (concise)
        if tier1.get('price_bucket'):
            parts.append(f"Price Range: {tier1['price_bucket']}")
        
        # Rating (concise)
        if tier1.get('avg_rating'):
            parts.append(f"Average Rating: {tier1['avg_rating']}/5.0 ({tier1.get('num_reviews', 0)} reviews)")
        
        # Business types (concise - only show top 2)
        if tier1.get('types') and len(tier1['types']) > 0:
            types_str = ', '.join(str(t).replace('_', ' ').title() for t in tier1['types'][:2])
            parts.append(f"Type: {types_str}")
        
        # Build metadata text
        metadata_text = "[RESTAURANT INFO]\\n" + "\\n".join(parts) + "\\n[END RESTAURANT INFO]"
        
        # Only add temporal context if hours are VALID
        if has_valid_hours and hours_dict:
            temporal_context = compute_temporal_context(tier1)
            if temporal_context:
                metadata_text += f"\\n\\n{temporal_context}"
        
        return metadata_text
    
    def enrich_with_tier1_features(self, results: List[Dict]) -> List[Dict]:
        """
        Enrich search results with Tier 1 features from enhanced DataFrame.
        Makes features visible in the text the LLM sees.
        
        CRITICAL FIX: Skip enrichment if hours_dict is invalid (all nulls)
        """
        if not self.tier1_enabled:
            return results
        
        enriched = []
        
        for result in results:
            restaurant_id = result.get('restaurant_id')
            
            if restaurant_id and restaurant_id in self.restaurant_lookup:
                row = self.restaurant_lookup[restaurant_id]
                
                # Check if hours_dict is valid
                hours_dict = row.get('hours_dict')
                has_valid_hours = False
                
                if isinstance(hours_dict, dict) and hours_dict:
                    # Check if at least one day has non-null hours
                    has_valid_hours = any(
                        v is not None and v != '' and pd.notna(v)
                        for v in hours_dict.values()
                    )
                
                # Add Tier 1 features using pre-processed columns
                tier1 = {
                    'hours_category': row.get('hours_category'),
                    'hours_dict': row.get('hours_dict') if has_valid_hours else None,  # ← Only include if valid
                    'hours_pretty': row.get('hours_pretty') if has_valid_hours else None,
                    'has_delivery': row.get('has_delivery'),
                    'has_takeout': row.get('has_takeout'),
                    'has_dine_in': row.get('has_dine_in'),
                    'price_level': row.get('price_level_numeric'),
                    'price_bucket': row.get('price_bucket'),
                    'types': row.get('types_normalized', [])[:5] if isinstance(row.get('types_normalized'), list) else [],
                    'avg_rating': row.get('avg_rating'),          
                    'num_reviews': row.get('num_reviews'),        
                    'address': row.get('address'),
                    'phone': row.get('phone'),
                }
                
                result['tier1_features'] = tier1
                
                # CRITICAL: Only prepend metadata if we have valid data to show
                # If hours_dict is invalid, let Qdrant's original data show through
                if has_valid_hours or tier1.get('address') or tier1.get('phone'):
                    # We have SOME valid metadata to add
                    metadata_text = self._format_metadata_text(tier1, result['restaurant'])
                    
                    # Prepend metadata to the full review text
                    if result.get('full_review'):
                        result['full_review'] = metadata_text + "\\n\\n" + result['full_review']
                    else:
                        result['full_review'] = metadata_text
                    
                    # Also add to chunk_text (first 500 chars shown in some views)
                    if result.get('chunk_text'):
                        metadata_preview = metadata_text[:300]
                        result['chunk_text'] = metadata_preview + "\\n\\n" + result['chunk_text'][:200]
                    else:
                        result['chunk_text'] = metadata_text[:500]
                else:
                    # No valid metadata - don't prepend anything
                    # Let Qdrant's original data be the only source
                    print(f"⚠️  Skipping enrichment for {result['restaurant']} - no valid metadata in parquet")
            
            enriched.append(result)
        
        return enriched
    
    # ========================================
    # LOCATION - AWARE SEARCH FUNCTIONS
    # ========================================
    def get_restaurants_within_radius(self, user_lat: float, user_lon: float, 
                                 radius_km: float = 5.0, limit: int = 100):
        """
        Get restaurants within radius with proper Phoenix tracing
        This now creates the correct span structure that Phoenix evaluators expect!
        """
        
        # Create Phoenix-compatible retrieve span
        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            span.set_attribute("tool.name", "geo_search")
            span.set_attribute("input.query", f"radius search {radius_km}km")
            span.set_attribute("retrieval.method", "qdrant_geo_filter")
            span.set_attribute("geo.user_lat", user_lat)
            span.set_attribute("geo.user_lon", user_lon)
            span.set_attribute("geo.radius_km", radius_km)

            span.set_attribute("user.location.enabled", True)
            span.set_attribute("user.location.city", "Christchurch")
            span.set_attribute("user.location.country", "New Zealand")

            try:
                # Calculate bounding box for efficient filtering 
                lat_delta = radius_km / 111.0 # ~111 km per degree latitude
                lon_delta = radius_km / (111.0 * math.cos(math.radians(user_lat)))

                # Create Qdrant geo filter
                geo_filter = Filter(
                    must=[
                        FieldCondition(
                            key="lat",
                            range=Range(gte=user_lat - lat_delta, lte=user_lat + lat_delta)
                        ),
                        FieldCondition(
                            key="lon", 
                            range=Range(gte=user_lon - lon_delta, lte=user_lon + lon_delta)
                        )
                    ]
                )
                
                # Search with geo filter
                results = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=geo_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                
                # Calculate exact distances and filter
                nearby_restaurants = []
                for point in results[0]:
                    payload = point.payload
                    
                    # Skip if no coordinates
                    if not payload.get('lat') or not payload.get('lon'):
                        continue
                        
                    # Calculate exact distance
                    distance = haversine_distance(
                        user_lat, user_lon,
                        float(payload['lat']), float(payload['lon'])
                    )
                    
                    # Filter by exact radius
                    if distance <= radius_km:
                        # Use existing _format_result method
                        result = self._format_result(point.id, 1.0)
                        if result:
                            result['distance_km'] = round(distance, 2)
                            result['distance_category'] = get_distance_category(distance)
                            nearby_restaurants.append(result)
                
                # Sort by distance
                nearby_restaurants.sort(key=lambda x: x['distance_km'])
                
                # Log retrieved documents for Phoenix evaluation
                for i, result in enumerate(nearby_restaurants):
                    prefix = f"retrieval.documents.{i}.document"
                    span.set_attribute(f"{prefix}.id", result['_id'])
                    span.set_attribute(f"{prefix}.restaurant", result['restaurant'])
                    span.set_attribute(f"{prefix}.score.distance", 1.0 - (result['distance_km'] / radius_km))
                    span.set_attribute(f"{prefix}.content", result['full_review'])
                    span.set_attribute(f"{prefix}.distance_km", result['distance_km'])
                    span.set_attribute(f"{prefix}.cuisines", json.dumps(result['cuisines']))
                
                span.set_attribute("retrieval.documents.count", len(nearby_restaurants))
                
                return nearby_restaurants
                
            except Exception as e:
                print(f"Error in geo search: {e}")
                span.record_exception(e)
                return []
    
    def enhance_smart_restaurant_search_with_location(self, query: str, 
                                                 user_lat: Optional[float] = None,
                                                 user_lon: Optional[float] = None):
        
        """
        Enhanced version of smart_restaurant_search that handles location queries.
        Replace or modify your existing smart_restaurant_search method.
        """

        # Detect location intent 
        location_intent = detect_location_intent(query)

        # Check if location query but no coordinates 
        if location_intent['is_location_query'] and (user_lat is None or user_lon is None):
            return {
                "mode": "location_required",
                "message": "Location-based query detected, but user location not available. Please enable location access.",
                "location_intent": location_intent,
                "results": []
            }
        
        # Handle location - based quries
        if location_intent['is_location_query'] and user_lat is not None and user_lon is not None:
            print(f"Location-based query: '{query}' (radius: {location_intent['radius_km']}km)")
            
            # Get restaurants within radius
            nearby_results = self.get_restaurants_within_radius(
                user_lat=user_lat,
                user_lon=user_lon,
                radius_km=location_intent['radius_km'],
                limit=100
            )

            # Apply existing cuisine detection and filtering 
            detected_cuisines = self.detect_cuisines_from_query(query)
            if detected_cuisines:
                # Filter by cuisine using your existing method
                nearby_results = [
                    r for r in nearby_results 
                    if any(cuisine in r.get('cuisines', []) for cuisine in detected_cuisines)
                ]
            
            # Apply your existing negative filters if needed
            negative_filters = self._extract_negative_filters(query) if hasattr(self, '_extract_negative_filters') else {}
            if negative_filters and any(negative_filters.values()):
                nearby_results = self._apply_negative_filters(nearby_results, negative_filters)

            # Deduplicate and limit results
            nearby_results = self.deduplicate_results(nearby_results)
            final_results = nearby_results[:5]

            # Enrich with Tier 1 features if available
            if self.tier1_enabled and final_results:
                final_results = self.enrich_with_tier1_features(final_results)
            
            return {
                "mode": "location_aware",
                "location_intent": location_intent,
                "detected_cuisines": detected_cuisines,
                "user_location": {"lat": user_lat, "lon": user_lon},
                "location_metadata": {
                    "enable": True,
                    "latitude": user_lat,
                    "longitude": user_lon,
                    "coordinates": f"{round(user_lat,2)}, {round(user_lon,2)}",
                    "city": "Christchurch",
                    "country": "New Zealand"
                    },
                "results": final_results,
                "total_nearby": len(nearby_results),
                "message": f"Found {len(final_results)} restaurants within {location_intent['radius_km']}km of your location"
            }
        
        # Fall back to existing smart_restaurant_search for non - location queries
        else:
            # Call your existing method
            return self.smart_restaurant_search(query)

    # ========================================
    # INITIALIZATION HELPERS
    # ========================================
        
    def _extract_negative_filters(self, query: str) -> Dict[str, List[str]]:
        '''Extract what user wants to AVOID.'''
        import re
        
        query_lower = query.lower()
        
        negative_terms = []
        
        # Handle "not too X" specially (e.g., "not too crowded")
        not_too_pattern = r'not\s+too\s+(\w+)'
        not_too_matches = re.findall(not_too_pattern, query_lower)
        for match in not_too_matches:
            # Only add the actual negative term, not "too"
            negative_terms.append(match)
        
        # Standard negative patterns (but skip if already matched by "not too")
        # Remove the matched "not too X" phrases to avoid double-matching
        cleaned_query = re.sub(not_too_pattern, '', query_lower)

        # Negative patterns
        patterns = [
            r'no\s+(\w+)',           # "no spicy"
            r'not\s+(\w+)',          # "not spicy"
            r'without\s+(\w+)',      # "without spicy"
            r'avoid\s+(\w+)',        # "avoid spicy"
            r"don't\s+want\s+(\w+)", # "don't want spicy"
            r'except\s+(\w+)',       # "except spicy"
            r'but\s+not\s+(\w+)',    # "but not spicy"
            r'but\s+no\s+(\w+)',     # "but no spicy"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_query)
            negative_terms.extend(matches)
        
        # Remove common stopwords that shouldn't be filtered
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'too'}
        negative_terms = [term for term in negative_terms if term not in stopwords]

        # Categorize
        result = {
            "exclude_cuisines": [],
            "exclude_features": []
        }
        
        cuisine_map = {
            'spicy': ['Indian', 'Thai', 'Mexican', 'Korean', 'Sichuan'],
            'seafood': ['Japanese', 'Seafood', 'Sushi'],
            'meat': ['BBQ', 'Steakhouse', 'Brazilian'],
        }
        
        for term in negative_terms:
            # Check if it's a cuisine keyword
            if term in cuisine_map:
                result["exclude_features"].append(term)
            # Check if direct cuisine
            elif any(term in cuisine.lower() for cuisine in self.all_cuisines):
                result["exclude_cuisines"].append(term)
            else:
                result["exclude_features"].append(term)

        # Remove duplicates
        result["exclude_cuisines"] = list(set(result["exclude_cuisines"]))
        result["exclude_features"] = list(set(result["exclude_features"]))
        
        return result
    
    def _apply_negative_filters(self, documents: List[Dict], negative_filters: Dict) -> List[Dict]:
        '''Remove documents matching exclusion criteria.'''
        if not any(negative_filters.values()):
            return documents
        
        filtered = []
        excluded_count = 0
        
        for doc in documents:
            text_lower = doc['full_review'].lower()
            cuisines_lower = [c.lower() for c in doc.get('cuisines', [])]
            
            exclude = False
            exclude_reason = ""

            # Check cuisine exclusions
            for cuisine_term in negative_filters.get('exclude_cuisines', []):
                if any(cuisine_term in c for c in cuisines_lower):
                    exclude = True
                    exclude_reason = f"cuisine: {cuisine_term}"
                    break
            
            # Check feature exclusions
            if not exclude:
                for feature in negative_filters.get('exclude_features', []):
                    # For certain features, check context to avoid false positives
                    if feature in ['crowded', 'busy', 'packed', 'noisy', 'loud']:
                        # Look for negative contexts around the feature word
                        # e.g., "very crowded", "too crowded", "was crowded"
                        negative_contexts = [
                            f'very {feature}',
                            f'too {feature}',
                            f'really {feature}',
                            f'so {feature}',
                            f'extremely {feature}',
                            f'quite {feature}',
                            f'was {feature}',
                            f'always {feature}',
                            f'{feature} place',
                            f'{feature} restaurant',
                        ]
                        
                        if any(context in text_lower for context in negative_contexts):
                            exclude = True
                            exclude_reason = f"feature: {feature} (negative context)"
                            break
                    
                    elif feature in ['expensive', 'pricey', 'costly']:
                        # Check for price-related negative contexts
                        price_contexts = [
                            f'very {feature}',
                            f'too {feature}',
                            f'quite {feature}',
                            f'pretty {feature}',
                            f'really {feature}',
                            f'{feature} but',
                            f'{feature} though',
                        ]
                        
                        if any(context in text_lower for context in price_contexts):
                            exclude = True
                            exclude_reason = f"feature: {feature} (price concern)"
                            break
                    
                    else:
                        # For other features (like spicy, seafood), simple substring match
                        if feature in text_lower:
                            exclude = True
                            exclude_reason = f"feature: {feature}"
                            break
            
            if not exclude:
                filtered.append(doc)
            else:
                excluded_count += 1
                print(f"Excluded: {doc['restaurant']} ({exclude_reason})")
        
        if excluded_count > 0:
            print(f"Filtered out {excluded_count} documents based on exclusions")
        
        return filtered
            
   
    def _cache_results(self, query: str, results: List[Dict]):
        """
        Cache query results with restaurant-aware key.
        
        ALTERNATIVE STRATEGY: Use (normalized_query, restaurant_name) as key
        instead of embedding similarity. This is more precise.
        """
        # Detect restaurant
        restaurant = self.detect_restaurant_name(query)
        
        # Normalize query (lowercase, remove punctuation)
        normalized = query.lower().strip()
        normalized = re.sub(r'[^a-z0-9\\s]', '', normalized)
        normalized = ' '.join(normalized.split())  # Remove extra spaces
        
        # Create cache key: (normalized_query, restaurant_name)
        cache_key = (normalized, restaurant if restaurant else "")
        
        self.query_cache[cache_key] = (query, results, datetime.now())
        
        # Limit cache size
        if len(self.query_cache) > 100:
            oldest_key = min(
                self.query_cache.keys(),
                key=lambda k: self.query_cache[k][2]
            )
            del self.query_cache[oldest_key]    
    
    def _get_cached_results(self, query: str) -> Optional[List[Dict]]:
        """
        Get cached results using exact match on (normalized_query, restaurant).
        This is more precise than embedding similarity.
        """
        # Detect restaurant
        restaurant = self.detect_restaurant_name(query)
        
        # Normalize query
        normalized = query.lower().strip()
        normalized = re.sub(r'[^a-z0-9\\s]', '', normalized)
        normalized = ' '.join(normalized.split())
        
        # Create cache key
        cache_key = (normalized, restaurant if restaurant else "")
        
        # Check if exact match exists
        if cache_key in self.query_cache:
            cached_query, results, timestamp = self.query_cache[cache_key]
            
            # Check if fresh (< 1 hour)
            age = (datetime.now() - timestamp).total_seconds()
            if age < 3600:
                print(f"Cache HIT (exact match)! Query: '{cached_query}'")
                return results
            else:
                print(f"Cache EXPIRED: {age/60:.1f} minutes old")
                del self.query_cache[cache_key]
        
        return None


    def _cross_encoder_rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        '''Rerank documents using cross-encoder for maximum accuracy.'''
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc['full_review']] for doc in documents]
        
        # Score all pairs (this is the magic!)
        scores = self.cross_encoder.predict(pairs)
        
        # Combine and sort
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and return top-k
        reranked = []
        for doc, score in scored_docs[:top_k]:
            doc_copy = doc.copy()
            doc_copy['cross_encoder_score'] = float(score)
            doc_copy['score'] = float(score)  # Replace hybrid score
            reranked.append(doc_copy)
        
        return reranked


    def _build_restaurant_cuisine_index(self):
        """Build a mapping of restaurant -> cuisines for fast lookup"""
        self.restaurant_cuisines = {}
        for doc in self.doc_lookup.values():
            restaurant = doc['payload']['restaurant']
            cuisines = doc['payload'].get('cuisines', [])
            if restaurant not in self.restaurant_cuisines:
                self.restaurant_cuisines[restaurant] = set()
            self.restaurant_cuisines[restaurant].update(cuisines)
        
        # Convert sets to lists
        for restaurant in self.restaurant_cuisines:
            self.restaurant_cuisines[restaurant] = list(self.restaurant_cuisines[restaurant])

    def build_cuisine_map(self):
        """Build cuisine keyword map for detection"""
        self.cuisine_keywords = {
            "chinese": ["chinese", "china", "dim sum", "dumpling", "wonton", "hotpot", "noodles"],
            "japanese": ["japanese", "japan", "sushi", "ramen", "udon", "sashimi", "teriyaki"],
            "korean": ["korean", "korea", "kimchi", "bibimbap", "bulgogi", "kbbq", "bbq"],  
            "thai": ["thai", "thailand", "pad thai", "tom yum", "curry", "spicy"],  
            "vietnamese": ["vietnamese", "vietnam", "pho", "banh mi", "spring roll"],  
            "indian": ["indian", "india", "curry", "tandoori", "biryani", "naan", "spicy"],  
            "italian": ["italian", "italy", "pizza", "pasta", "spaghetti", "lasagna"], 
            "mexican": ["mexican", "mexico", "taco", "burrito", "quesadilla"],  
            "american": ["american", "burger", "bbq", "steak", "fries"],
            "bar": ["bar", "pub", "tavern", "brewery", "cocktail", "drinks"],
            "cafe": ["cafe", "coffee", "espresso", "latte", "cappuccino"],
            "bakery": ["bakery", "bread", "pastry", "croissant"],
            "fast_food": ["fast food", "drive thru", "quick service"],
        }
        
        self.keyword_to_cuisine = {}
        for cuisine, keywords in self.cuisine_keywords.items():
            for keyword in keywords:
                self.keyword_to_cuisine[keyword] = cuisine

    # ========================================
    # ENHANCED QUERY UNDERSTANDING
    # ========================================

    def classify_query_intent(self, query: str) -> Dict[str, any]:
        """Classify query intent for better routing"""
        q = query.lower()
        
        intent = {
            'type': 'general',
            'needs_reranking': False,
            'is_comparative': False,
            'is_specific_dish': False,
            'is_location_based': False,
        }
        
        # Comparative queries (need multi-query expansion)
        comparative_keywords = ['best', 'top', 'better', 'vs', 'versus', 'compare', 'recommend', 'popular', 'good']
        if any(k in q for k in comparative_keywords):
            intent['type'] = 'comparative'
            intent['needs_reranking'] = True
            intent['is_comparative'] = True
        
        # Specific factual queries (direct hybrid search)
        factual_keywords = ['open', 'hours', 'address', 'phone', 'location', 'price', 'cost', 'how much']
        if any(k in q for k in factual_keywords):
            intent['type'] = 'factual'
            intent['needs_reranking'] = False
        
        # Dish-specific queries
        dish_patterns = [r'does .* have', r'serve', 'menu', 'dish', 'food', 'meal']
        if any(re.search(p, q) if p.startswith('r') else p in q for p in dish_patterns):
            intent['is_specific_dish'] = True
        
        # Location queries
        if any(word in q for word in ['near', 'location', 'where', 'address']):
            intent['is_location_based'] = True
        
        return intent

    def detect_cuisines_from_query(self, query: str) -> List[str]:
        """Enhanced cuisine detection with semantic expansion"""
        query_lower = query.lower()
        detected_cuisines = set()
        
        # Direct keyword matching
        for keyword, cuisine in self.keyword_to_cuisine.items():
            if keyword in query_lower:
                for tag in self.all_cuisines:
                    tag_lower = tag.lower()
                    if cuisine in tag_lower or tag_lower in cuisine:
                        detected_cuisines.add(tag)
        
        # Exact cuisine tag matching
        for cuisine_tag in self.all_cuisines:
            if cuisine_tag.lower() in query_lower:
                detected_cuisines.add(cuisine_tag)
        
        # Semantic expansion for ambiguous terms
        semantic_mappings = {
            'asian': ['Chinese', 'Japanese', 'Korean', 'Thai', 'Vietnamese'],
            'western': ['American', 'Italian', 'French', 'Mediterranean'],
            'spicy': ['Thai', 'Indian', 'Korean', 'Mexican'],
            'noodles': ['Japanese', 'Chinese', 'Vietnamese', 'Thai'],
            'bbq': ['Korean', 'American', 'BBQ'],
            'fast food': ['American', 'Burger'],
            'seafood': ['Japanese', 'Mediterranean'],
        }
        
        for term, cuisine_list in semantic_mappings.items():
            if term in query_lower:
                for cuisine in cuisine_list:
                    for tag in self.all_cuisines:
                        if cuisine.lower() in tag.lower():
                            detected_cuisines.add(tag)
        
        return sorted(list(detected_cuisines))

    def detect_restaurant_name(self, query: str) -> Optional[str]:
        """
        Enhanced restaurant name detection with LONGEST MATCH FIRST
        
        CRITICAL FIX: Sort restaurants by length (longest first) to avoid
        partial matches. "Lone Star Spitfire" must be checked BEFORE "Lone Star".
        
        Features:
        - Word boundary matching
        - Query cleaning
        - Stopword filtering
        - Suffix normalization
        - Cuisine validation
        - Longest match priority (NEW!)
        """
        q = query.lower().replace("'", "").strip()

        # Check if this is a pure cuisine query (should NOT detect restaurants)
        pure_cuisine_keywords = {
            'sushi', 'pizza', 'chinese', 'italian', 'thai', 'indian', 'korean', 
            'vietnamese', 'mexican', 'japanese', 'american', 'french', 'greek',
            'seafood', 'bbq', 'burgers', 'pasta', 'ramen', 'noodles', 'curry',
            'tacos', 'burritos', 'steaks', 'fish', 'chicken'
        }
        
        # If query is ONLY a cuisine keyword, don't detect restaurants
        if q in pure_cuisine_keywords:
            print(f"Pure cuisine query detected: '{q}' - skipping restaurant detection")
            return None

        # Check for cuisine context words
        cuisine_context_words = {
            'best', 'good', 'great', 'top', 'recommend', 'places', 'restaurants',
            'nearby', 'near me', 'around', 'area', 'find', 'looking for',
            'where', 'any', 'some', 'options', 'suggestions', 'cheap', 'expensive'
        }

        has_cuisine_context = any(word in q for word in cuisine_context_words)

        if has_cuisine_context:
            print(f"Cuisine context detected - skipping restaurant name detection")
            return None

        # Remove suffixes safely
        suffixes = [" restaurant", " cafe", " bar", " bistro", " eatery"]
        q_normalized = q
        for suf in suffixes:
            if q_normalized.endswith(suf):
                q_normalized = q_normalized.replace(suf, "")

        # STOPWORDS - EXPANDED
        stopwords = {
            "best", "good", "top", "great", "recommend", "find", "where",
            "in", "at", "the", "a", "an", "near", "christchurch",
            "new", "zealand", "food", "place", "open", "close",
            "time", "hour", "does", "is", "yet", "now", "what", "when"
        }

        # Filter stopwords from query
        q_tokens = [t for t in q.split() if t not in stopwords]
        q_filtered = " ".join(q_tokens)

        detected_cuisines = self.detect_cuisines_from_query(query)

        # Sort restaurants by length (LONGEST FIRST)
        # This ensures "lone star spitfire" is checked before "lone star"
        sorted_restaurants = sorted(
            self.all_restaurants,
            key=lambda r: len(r),
            reverse=True  # Longest first!
        )

        # MAIN MATCHING (now using sorted list)
        for restaurant in sorted_restaurants:
            r_norm = restaurant.lower().replace("'", "")

            # Remove suffixes from restaurant names too
            r_clean = r_norm
            for suf in suffixes:
                if r_clean.endswith(suf):
                    r_clean = r_clean.replace(suf, "")

            # 1) Exact match or substring match
            if r_norm in q or r_norm in q_filtered or r_clean in q_filtered:
                print(f"🎯 Detected restaurant (exact): '{restaurant}'")
                return restaurant

            # 2) Word boundary matching (SAFER)
            pattern = r"\\b" + re.escape(r_clean) + r"\\b"
            if re.search(pattern, q_filtered):
                # CUISINE VALIDATION
                if detected_cuisines:
                    cuisines = self.restaurant_cuisines.get(restaurant, [])
                    if not any(c in cuisines for c in detected_cuisines):
                        continue
                print(f"🎯 Detected restaurant (word boundary): '{restaurant}'")
                return restaurant

        return None
    
    # ========================================
    # CACHING FOR PERFORMANCE
    # ========================================

    @lru_cache(maxsize=200)
    def _cached_vector_encode(self, text: str) -> Tuple:
        """Cache embeddings for repeated queries"""
        return tuple(self.model.encode(text).tolist())

    # ========================================
    # CORE SEARCH METHODS
    # ========================================

    def _bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """BM25 keyword search with improved tokenization"""
        # Tokenize query same way as corpus
        stop_words = get_stopwords()
        query_clean = query.translate(str.maketrans('', '', string.punctuation))
        tokenized_query = [
            word.lower() for word in query_clean.split()
            if word.lower() not in stop_words and len(word) > 2
        ]
        
        if not tokenized_query:  # Fallback to simple split if no tokens
            tokenized_query = query.lower().split()
        
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = self.bm25_doc_ids[idx]
            score = float(scores[idx])
            if score > 0:
                results.append((doc_id, score))
        
        return results
    
    def _bm25_search_filtered(self, query: str, top_k: int = 20, 
                         restaurant: Optional[str] = None) -> List[Tuple[str, float]]:
        """Optimized BM25 search with restaurant pre-filtering"""
        
        try:
            # Tokenize query
            stop_words = get_stopwords()
            query_clean = query.translate(str.maketrans('', '', string.punctuation))
            tokenized_query = [
                word.lower() for word in query_clean.split()
                if word.lower() not in stop_words and len(word) > 2
            ]
            
            if not tokenized_query:
                tokenized_query = query.lower().split()
            
            if restaurant:
                restaurant_normalized = self._normalize_restaurant_name(restaurant)
                
                # Use pre-built index
                valid_indices = self.restaurant_bm25_index.get(restaurant_normalized, [])
                
                if not valid_indices:
                    print(f"   ⚠️  No documents found in BM25 index for restaurant: '{restaurant}'")
                    return []
                
                # Score only valid documents
                scores = self.bm25.get_scores(tokenized_query)
                
                results = []
                for idx in valid_indices:
                    score = float(scores[idx])
                    if score > 0:
                        doc_id = self.bm25_doc_ids[idx]
                        results.append((doc_id, score))
                
                results.sort(key=lambda x: x[1], reverse=True)
                print(f"   ✅ BM25 filtered: {len(results)} docs from '{restaurant}'")
                return results[:top_k]
            
            else:
                # Original unfiltered BM25
                scores = self.bm25.get_scores(tokenized_query)
                top_indices = np.argsort(scores)[::-1][:top_k]
                
                results = []
                for idx in top_indices:
                    doc_id = self.bm25_doc_ids[idx]
                    score = float(scores[idx])
                    if score > 0:
                        results.append((doc_id, score))
                
                return results
        
        except Exception as e:
            print(f"❌ Error in BM25 filtered search: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to regular BM25
            print(f"   → Falling back to unfiltered BM25")
            return self._bm25_search(query, top_k)

    def _vector_search(self, query: str, top_k: int = 20,
                      filters: Optional[Filter] = None) -> List[Tuple[str, float]]:
        """Vector semantic search with caching"""
        query_vector = list(self._cached_vector_encode(query))
        
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=filters
        )
        
        return [(hit.id, hit.score) for hit in results.points]

    def _reciprocal_rank_fusion(self, rankings: List[List[Tuple]], 
                                k: int = 60) -> List[Tuple[str, float]]:
        """Combine rankings using Reciprocal Rank Fusion"""
        fused_scores = {}
        
        for ranking in rankings:
            for rank, (doc_id, score) in enumerate(ranking, 1):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank)
        
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def _apply_filters(self, results: List[Tuple[str, float]], cuisines=None, 
                      restaurant=None, min_rating=None) -> List[Tuple[str, float]]:
        """Apply filters to BM25 results - NOW STRICT FOR RESTAURANTS!"""
        if not (cuisines or restaurant or min_rating):
            return results
        
        filtered = []
        for doc_id, score in results:
            doc = self.doc_lookup.get(doc_id)
            if not doc:
                continue
            
            payload = doc['payload']

            # If restaurant specified, ONLY return exact matches
            if restaurant:
                if payload.get('restaurant') != restaurant:
                    continue  # Skip this document entirely
            
            if cuisines:
                doc_cuisines = payload.get('cuisines', [])
                if not any(c in doc_cuisines for c in cuisines):
                    continue
            
            if min_rating and payload.get('rating', 0) < min_rating:
                continue
            
            filtered.append((doc_id, score))
        
        return filtered

    def _build_qdrant_filter(self, cuisines=None, restaurant=None, 
                        min_rating=None, restaurant_id=None) -> Optional[Filter]:
        """Build Qdrant filter from parameters"""
        conditions = []

        # Restaurant filter by ID (MOST RELIABLE)
        if restaurant_id:
            conditions.append(
                FieldCondition(key="restaurant_id", match=MatchValue(value=restaurant_id))
            )
            print(f"🎯 Added STRICT restaurant_id filter: '{restaurant_id}'")
        # Fallback to name-based filter if no ID
        elif restaurant:
            conditions.append(
                FieldCondition(key="restaurant", match=MatchValue(value=restaurant))
            )
            print(f"🎯 Added restaurant name filter: '{restaurant}'")
        
        if cuisines:
            conditions.append(
                FieldCondition(key="cuisines", match=MatchAny(any=cuisines))
            )
        
        if min_rating:
            conditions.append(
                FieldCondition(key="rating", range=Range(gte=min_rating))
            )
        
        return Filter(must=conditions) if conditions else None

    def _format_result(self, doc_id: str, score: float) -> Optional[Dict]:
        """Convert doc_id to formatted result"""
        doc = self.doc_lookup.get(doc_id)
        if not doc:
            return None
        
        payload = doc['payload']
        
        result = {
            '_id': doc_id,
            'score': score,
            'restaurant': payload['restaurant'],
            'restaurant_id': payload.get('restaurant_id'),
            'cuisines': payload.get('cuisines', []),
            'full_review': payload['text'],
            'chunk_text': payload['text'][:500],
            'type': payload['type']
        }
        
        if payload['type'] == 'review':
            result['rating'] = payload.get('rating')
            result['platform'] = payload.get('platform', '')
            result['user'] = payload.get('user', '')
        elif payload['type'] == 'restaurant_summary':
            result['avg_rating'] = payload.get('avg_rating')
            result['num_reviews'] = payload.get('num_reviews')
        
        return result

    def _diversify_results(self, results: List[Dict], max_per_restaurant: int = 2) -> List[Dict]:
        """Ensure diversity in results - no more than N reviews per restaurant"""
        diversified = []
        restaurant_counts = {}
        
        for result in results:
            restaurant = result['restaurant']
            count = restaurant_counts.get(restaurant, 0)
            
            if count < max_per_restaurant:
                diversified.append(result)
                restaurant_counts[restaurant] = count + 1
        
        return diversified

    # ========================================
    # HIGH-LEVEL SEARCH INTERFACE
    # ========================================

    def hybrid_search(self, query: str, top_k: int = 5,
                 cuisines: Optional[List[str]] = None,
                 restaurant: Optional[str] = None,
                 min_rating: Optional[float] = None,
                 apply_diversity: bool = True,
                 qdrant_filter: Optional[Filter] = None) -> List[Dict]:
        """
        Main hybrid search: BM25 + Vector with RRF fusion.
        Enhanced with fallback logic, diversity, and Tier 1 filtering.
        FIXED: Restaurant-specific queries now properly filter at retrieval time.
        
        Args:
            query: Search query
            top_k: Number of results to return
            cuisines: Filter by cuisines
            restaurant: Filter by specific restaurant
            min_rating: Minimum rating filter
            apply_diversity: Limit results per restaurant (DISABLED for restaurant-specific queries)
            qdrant_filter: Optional Tier 1 Qdrant Filter object
        """
        
        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            span.set_attribute("tool.name", "hybrid_search")
            span.set_attribute("input.query", query)
            span.set_attribute("retrieval.method", "hybrid_bm25_vector")
            
            if cuisines:
                span.set_attribute("filter.cuisines", json.dumps(cuisines))
            if restaurant:
                span.set_attribute("filter.restaurant", restaurant)
                print(f"🎯 RESTAURANT-SPECIFIC QUERY: '{restaurant}'")
            if min_rating:
                span.set_attribute("filter.min_rating", min_rating)
            if qdrant_filter:
                span.set_attribute("filter.tier1_enabled", True)
            
            # ========================================
            # STEP 1: BUILD FILTERS
            # ========================================

            # Look up restaurant_id if restaurant is specified
            restaurant_id = None
            if restaurant and self.tier1_enabled and self.restaurant_lookup:
                # Find restaurant_id from restaurant name
                restaurant_id = next(
                    (
                        r_id
                        for r_id, data in self.restaurant_lookup.items()
                        if self._normalize_restaurant_name(data.get("restaurant", "")) == 
                        self._normalize_restaurant_name(restaurant)
                    ),
                    None,
                )
                if restaurant_id:
                    print(f"   ✅ Found restaurant_id: '{restaurant_id}' for '{restaurant}'")
                else:
                    print(f"   ⚠️  No restaurant_id found for '{restaurant}' - using name fallback")

            # Build standard filter (NOW WITH restaurant_id)
            standard_filter = self._build_qdrant_filter(
                cuisines, restaurant, min_rating, restaurant_id=restaurant_id
            )

            # If restaurant specified but no filter created, create one
            if restaurant and not standard_filter:
                if restaurant_id:
                    standard_filter = Filter(
                        must=[FieldCondition(key="restaurant_id", match=MatchValue(value=restaurant_id))]
                    )
                else:
                    standard_filter = Filter(
                        must=[FieldCondition(key="restaurant", match=MatchValue(value=restaurant))]
                    )

            # Use Tier 1 filter if provided, otherwise use standard filter
            final_filter = qdrant_filter if qdrant_filter else standard_filter

            # ✅ CRITICAL: Add restaurant filter to Tier 1 filter if restaurant specified
            if restaurant and qdrant_filter:
                # Use restaurant_id if available, otherwise use name
                if restaurant_id:
                    if not any(c.key == "restaurant_id" for c in qdrant_filter.must):
                        qdrant_filter.must.append(
                            FieldCondition(key="restaurant_id", match=MatchValue(value=restaurant_id))
                        )
                else:
                    if not any(c.key == "restaurant" for c in qdrant_filter.must):
                        qdrant_filter.must.append(
                            FieldCondition(key="restaurant", match=MatchValue(value=restaurant))
                        )
                final_filter = qdrant_filter
            
            # ========================================
            # STEP 2: RETRIEVE FROM BOTH SOURCES
            # ========================================
            
            # ✅ UPDATED: Use filtered BM25 for restaurant queries
            if restaurant:
                # Use filtered BM25 to prevent partial matches
                bm25_results = self._bm25_search_filtered(query, top_k=20, restaurant=restaurant)
            else:
                # Use regular BM25 for non-restaurant queries
                bm25_results = self._bm25_search(query, top_k=20)

            vector_results = self._vector_search(query, top_k=20, filters=final_filter)

            # ✅ CRITICAL: Apply filters to BM25 STRICTLY (only if not already filtered)
            if not restaurant:
                bm25_results = self._apply_filters(
                    bm25_results, cuisines, restaurant, min_rating
                )

            # Debug output
            if restaurant:
                print(f"   → BM25 returned {len(bm25_results)} results")
                print(f"   → Vector returned {len(vector_results)} results")
            
            # ========================================
            # STEP 3: FUSE RANKINGS
            # ========================================
            
            fused_results = self._reciprocal_rank_fusion([bm25_results, vector_results])
            
            # ========================================
            # STEP 4: FORMAT RESULTS
            # ========================================
            
            # For restaurant-specific queries, get more candidates to ensure we have enough
            candidate_limit = top_k * 5 if restaurant else top_k * 3
            
            candidate_results = []
            for doc_id, score in fused_results[:candidate_limit]:
                result = self._format_result(doc_id, score)
                if result:
                    candidate_results.append(result)
            
            # ========================================
            # STEP 5: VALIDATE FIRST 
            # ========================================

            if restaurant and candidate_results:
                # ✅ VALIDATE BEFORE LIMITING - ensures we get ALL results from target restaurant
                print(f"   → Validating {len(candidate_results)} candidates for '{restaurant}'...")
                
                restaurant_normalized = self._normalize_restaurant_name(restaurant)
                
                valid_results = []
                invalid_results = []
                
                # Filter ALL candidates, not just top_k
                for result in candidate_results:
                    result_restaurant = result.get('restaurant', '')
                    result_normalized = self._normalize_restaurant_name(result_restaurant)
                    
                    if result_normalized == restaurant_normalized:
                        valid_results.append(result)
                    else:
                        invalid_results.append(result)
                
                # Log what was removed
                if invalid_results:
                    print(f"   ⚠️  REMOVED {len(invalid_results)} wrong restaurants:")
                    for r in invalid_results[:3]:  # Show first 3
                        print(f"      - {r.get('restaurant', 'Unknown')}")
                
                # NOW take top_k from validated results
                final_results = valid_results[:top_k]
                print(f"   ✅ Found {len(valid_results)} total from '{restaurant}', returning top {min(top_k, len(valid_results))}")

            elif apply_diversity and not restaurant:
                # Apply diversity for non-restaurant queries
                final_results = self._diversify_results(candidate_results, max_per_restaurant=2)
                final_results = final_results[:top_k]
            else:
                final_results = candidate_results[:top_k]

            # ========================================
            # STEP 6: FALLBACK HANDLING  
            # ========================================

            if not final_results:
                if cuisines and not restaurant:
                    span.add_event("No results with cuisine filter, retrying without cuisines")
                    return self.hybrid_search(
                        query, top_k, cuisines=None, restaurant=restaurant, 
                        min_rating=min_rating, apply_diversity=apply_diversity
                    )
                elif min_rating and not restaurant:
                    span.add_event("No results with rating filter, retrying without rating")
                    return self.hybrid_search(
                        query, top_k, cuisines=cuisines, restaurant=restaurant, 
                        min_rating=None, apply_diversity=apply_diversity
                    )
                elif restaurant:
                    print(f"   ⚠️  No results found for '{restaurant}'")
                    span.add_event(f"No results found for restaurant: {restaurant}")

            # ========================================
            # STEP 7: PHOENIX LOGGING
            # ========================================

            for i, result in enumerate(final_results):
                prefix = f"retrieval.documents.{i}.document"
                span.set_attribute(f"{prefix}.id", result['_id'])
                span.set_attribute(f"{prefix}.restaurant", result['restaurant'])
                span.set_attribute(f"{prefix}.score.hybrid", result['score'])
                span.set_attribute(f"{prefix}.content", result['full_review'])
                span.set_attribute(f"{prefix}.cuisines", json.dumps(result['cuisines']))
                
                # Log Tier 1 metadata if available
                if result.get('tier1_features'):
                    tier1 = result['tier1_features']
                    
                    if tier1.get('address'):
                        span.set_attribute(f"{prefix}.metadata.address", tier1['address'])
                    if tier1.get('phone'):
                        span.set_attribute(f"{prefix}.metadata.phone", tier1['phone'])
                    if tier1.get('hours_pretty'):
                        span.set_attribute(f"{prefix}.metadata.hours", tier1['hours_pretty'])
                    if tier1.get('price_bucket'):
                        span.set_attribute(f"{prefix}.metadata.price", tier1['price_bucket'])
                    if tier1.get('avg_rating') is not None:
                        span.set_attribute(f"{prefix}.metadata.rating", tier1['avg_rating'])
                    if tier1.get('has_delivery') is not None:
                        span.set_attribute(f"{prefix}.metadata.has_delivery", tier1['has_delivery'])
                    if tier1.get('has_takeout') is not None:
                        span.set_attribute(f"{prefix}.metadata.has_takeout", tier1['has_takeout'])
                    if tier1.get('has_dine_in') is not None:
                        span.set_attribute(f"{prefix}.metadata.has_dine_in", tier1['has_dine_in'])

            span.set_attribute("retrieval.documents.count", len(final_results))

            # ✅ Log validation metrics
            if restaurant:
                all_correct = all(
                    self._normalize_restaurant_name(r.get('restaurant', '')) == 
                    self._normalize_restaurant_name(restaurant) 
                    for r in final_results
                )
                span.set_attribute("retrieval.restaurant_filter_strict", True)
                span.set_attribute("retrieval.all_from_correct_restaurant", all_correct)

            return final_results

    # ========================================
    # BACKWARD COMPATIBILITY
    # ========================================

    def search_reviews(self, query, restaurant=""):
        """Backward-compatible simple search"""
        return self.hybrid_search(
            query=query,
            restaurant=restaurant if restaurant else None,
            top_k=5
        )

    def search_reviews_with_cuisine(self, query, cuisines=None, restaurant=""):
        """Backward-compatible cuisine search"""
        return self.hybrid_search(
            query=query,
            cuisines=cuisines,
            restaurant=restaurant if restaurant else None,
            top_k=5
        )

    # ========================================
    # ADVANCED SEARCH METHODS
    # ========================================

    def expand_queries(self, query: str) -> List[str]:
        """Query expansion using GPT"""
        prompt = f"""
Rewrite the user's query into 5 diverse search queries.
These queries must capture different aspects and wording.
Do not introduce new ideas.

User query: "{query}"

Return ONLY JSON like:
{{
  "queries": [
    "query 1",
    "query 2",
    "query 3",
    "query 4",
    "query 5"
  ]
}}
"""
        resp = self.openai_client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )

        raw = resp.output[0].content[0].text

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])

        return data["queries"]

    def search_reviews_reranked_multiples_queries(
        self, query: str, restaurant: str = "",
        expanded_queries: list = None, cuisines: list = None,
        qdrant_filter: Optional[Filter] = None
    ):
        """Multi-query + HyDE reranking with optional Tier 1 filtering"""
        
        if expanded_queries is None:
            expanded_queries = self.expand_queries(query)
        
        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            span.set_attribute("tool.name", "multi_query_reranked")
            span.set_attribute("input.query", query)
            span.set_attribute("retrieval.expanded_queries", json.dumps(expanded_queries))
            
            # Collect from all queries
            all_results = []
            for q in expanded_queries:
                results = self.hybrid_search(
                    query=q,
                    cuisines=cuisines,
                    restaurant=restaurant if restaurant else None,
                    top_k=10,
                    apply_diversity=False,  # Apply diversity after reranking
                    qdrant_filter=qdrant_filter  # Pass Tier 1 filter
                )
                all_results.extend(results)
            
            # Deduplicate
            seen = set()
            candidates = []
            for result in all_results:
                if result['_id'] not in seen:
                    seen.add(result['_id'])
                    candidates.append(result)
            
            if not candidates:
                return {
                    "expanded_queries": expanded_queries,
                    "cuisines_detected": cuisines or [],
                    "results": []
                }
            
            # HyDE reranking
            chunks = [r['full_review'] for r in candidates]
            
            hyde_prompt = f"""
You MUST answer only in valid JSON.

Task:
- Create a hypothetical ideal answer to the user's question.
- Do NOT use any real facts.
- Use placeholders like NAME, PLACE, DISH.

Return EXACTLY this JSON:
{{
"hypotheticalAnswer": "text"
}}

User question: "{query}"
"""
            hyde = self.openai_client.responses.create(
                model="gpt-4o-mini",
                input=hyde_prompt
            )

            hyde_text = hyde.output[0].content[0].text

            try:
                hyde_json = json.loads(hyde_text)
            except:
                start = hyde_text.find("{")
                end = hyde_text.rfind("}") + 1
                hyde_json = json.loads(hyde_text[start:end])

            hypothetical_answer = hyde_json["hypotheticalAnswer"]
            
            # Embeddings
            hyp_emb = np.array(
                self.openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=hypothetical_answer
                ).data[0].embedding
            )

            emb_chunks = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=chunks
            )

            chunk_embs = np.array([e.embedding for e in emb_chunks.data])
            
            # Ranking
            scores = chunk_embs @ hyp_emb
            ranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Apply diversity to top results
            top_results = [doc for doc, score in ranked[:15]]
            diverse_results = self._diversify_results(top_results, max_per_restaurant=2)[:5]
            
            # Rebuild with scores
            final_ranked = []
            for doc in diverse_results:
                for candidate, score in ranked:
                    if candidate['_id'] == doc['_id']:
                        final_ranked.append((doc, score))
                        break
            
            # Phoenix logging
            for i, (doc, score) in enumerate(final_ranked):
                prefix = f"retrieval.documents.{i}.document"
                span.set_attribute(f"{prefix}.id", doc['_id'])
                span.set_attribute(f"{prefix}.restaurant", doc['restaurant'])
                span.set_attribute(f"{prefix}.score.hybrid", float(score))
                span.set_attribute(f"{prefix}.content", doc['full_review'])
                span.set_attribute(f"{prefix}.cuisines", json.dumps(doc['cuisines']))
            
            span.set_attribute("retrieval.documents.count", len(final_ranked))
            
            return {
                "expanded_queries": expanded_queries,
                "cuisines_detected": cuisines or [],
                "results": [
                    {
                        "doc": {k: v for k, v in doc.items() if k != 'score'},
                        "score": float(score),
                        "cuisines": doc['cuisines']
                    }
                    for doc, score in final_ranked
                ]
            }

    # ========================================
    # SMART SEARCH - MAIN ENTRY POINT
    # ========================================

    def smart_restaurant_search(
        self,
        query: str,
        use_cache: bool = True,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        max_distance_km: Optional[float] = None
    ):
        start_time = time.time()

        # -------------------------------------------------
        # STEP 1: Location intent (SINGLE entry point)
        # -------------------------------------------------
        location_intent = detect_location_intent(query, user_max_distance_km=max_distance_km)

        if location_intent["is_location_query"]:
            if user_lat is None or user_lon is None:
                print("Location query detected but coordinates are None")
                print(f"   user_lat: {user_lat} ({type(user_lat)})")
                print(f"   user_lon: {user_lon} ({type(user_lon)})")

                return {
                    "mode": "location_required",
                    "message": "Location-based query detected, but user location not available.",
                    "location_intent": location_intent,
                    "results": [],
                }

            print(f"🗺️ Location query: radius={location_intent['radius_km']}km")

            nearby_results = self.get_restaurants_within_radius(
                user_lat=user_lat,
                user_lon=user_lon,
                radius_km=location_intent["radius_km"],
            )

            cuisines = self.detect_cuisines_from_query(query)
            if cuisines:
                nearby_results = [
                    r
                    for r in nearby_results
                    if any(c in r.get("cuisines", []) for c in cuisines)
                ]

            final_results = self.deduplicate_results(nearby_results)[:5]

            if self.tier1_enabled:
                final_results = self.enrich_with_tier1_features(final_results)

            return {
                "mode": "location_aware",
                "location_intent": location_intent,
                "detected_cuisines": cuisines,
                "results": final_results,
                "total_nearby": len(nearby_results),
            }

        # -------------------------------------------------
        # STEP 2: Cache
        # -------------------------------------------------
        if use_cache:
            cached = self._get_cached_results(query)
            if cached:
                elapsed = (time.time() - start_time) * 1000
                return {
                    "mode": "cached",
                    "results": cached,
                    "cache_hit": True,
                    "search_time_ms": round(elapsed, 2),
                }

        # -------------------------------------------------
        # STEP 3: Entity + intent detection
        # -------------------------------------------------
        restaurant = self.detect_restaurant_name(query)
        if restaurant:
            # IMPROVED: O(1) existence check with fuzzy suggestions
            restaurant_normalized = self._normalize_restaurant_name(restaurant)
            
            if restaurant_normalized not in self.normalized_restaurants:
                print(f"⚠️  Restaurant '{restaurant}' not found in database")
                
                # Fuzzy matching for suggestions
                from difflib import get_close_matches
                suggestions = get_close_matches(
                    restaurant_normalized,
                    self.normalized_restaurants.keys(),
                    n=5,
                    cutoff=0.6
                )
                # Convert back to original names
                suggestions = [self.normalized_restaurants[s] for s in suggestions]
                
                return {
                    "mode": "restaurant_not_found",
                    "restaurant_detected": restaurant,
                    "message": f"Sorry, I couldn't find '{restaurant}' in our database. Did you mean one of these?",
                    "results": [],
                    "suggestions": suggestions
                }
            
        cuisines = self.detect_cuisines_from_query(query)
        intent = self.classify_query_intent(query)

        tier1_intents = (
            self.detect_tier1_filter_intents(query) if self.tier1_enabled else {}
        )
        tier1_filter = self.apply_tier1_filters(tier1_intents) if tier1_intents else None

        if restaurant:
            print(f"🎯 Restaurant-specific query detected: '{restaurant}'")

            restaurant_filter = Filter(
                must=[FieldCondition(key="restaurant", match=MatchValue(value=restaurant))]
            )

            if tier1_filter:
                tier1_filter.must.append(restaurant_filter.must[0])
            else:
                tier1_filter = restaurant_filter

        negative_filters = (
            self._extract_negative_filters(query)
            if hasattr(self, "_extract_negative_filters")
            else {}
        )

        force_enrichment = restaurant is not None

        # -------------------------------------------------
        # STEP 4: retrieve_k logic 
        # -------------------------------------------------
        retrieve_k = 15
        if restaurant and self.tier1_enabled:
            restaurant_id = next(
                (
                    r_id
                    for r_id, data in self.restaurant_lookup.items()
                    if data.get("restaurant") == restaurant
                ),
                None,
            )
            if restaurant_id:
                num_reviews = self.restaurant_lookup[restaurant_id].get("num_reviews", 5)
                retrieve_k = min(15, max(5, num_reviews + 3))

        # -------------------------------------------------
        # STEP 5: Routing
        # -------------------------------------------------
        if restaurant:
            initial_results = self.hybrid_search(
                query,
                restaurant=restaurant,
                top_k=retrieve_k,
                qdrant_filter=tier1_filter,
            )
            mode_base = "direct_filtered"

        elif intent["type"] == "factual":
            initial_results = self.hybrid_search(
                query, cuisines=cuisines, top_k=retrieve_k, qdrant_filter=tier1_filter
            )
            mode_base = "factual_hybrid"

        elif intent["type"] == "comparative" or cuisines:
            result_data = self.search_reviews_reranked_multiples_queries(
                query, cuisines=cuisines, qdrant_filter=tier1_filter
            )
            initial_results = (
                [item["doc"] for item in result_data["results"][:retrieve_k]]
                if result_data.get("results")
                else []
            )
            mode_base = "reranked_comparative"

        else:
            initial_results = self.hybrid_search(
                query, top_k=retrieve_k, qdrant_filter=tier1_filter
            )
            mode_base = "fallback"

        # -------------------------------------------------
        # STEP 6–12: unchanged pipeline
        # -------------------------------------------------
        if negative_filters and any(negative_filters.values()) and initial_results:
            initial_results = self._apply_negative_filters(
                initial_results, negative_filters
            )

        initial_results = self.deduplicate_results(initial_results)

        if hasattr(self, "_cross_encoder_rerank") and initial_results:
            final_results = self._cross_encoder_rerank(query, initial_results, top_k=5)
            mode_suffix = "_cross_encoder_reranked"
        else:
            final_results = initial_results[:5]
            mode_suffix = ""

        if self.tier1_enabled and final_results and (force_enrichment or tier1_intents):
            final_results = self.enrich_with_tier1_features(final_results)

        if use_cache and final_results:
            self._cache_results(query, final_results)

        elapsed = (time.time() - start_time) * 1000

        result = {
            "mode": f"{mode_base}{mode_suffix}",
            "restaurant_detected": restaurant,
            "cuisines_detected": cuisines,
            "intent": intent,
            "results": final_results,
            "cache_hit": False,
            "search_time_ms": round(elapsed, 2),
            "enriched": force_enrichment or bool(tier1_intents),
        }

        if tier1_intents:
            result["tier1_filters"] = tier1_intents
        if negative_filters and any(negative_filters.values()):
            result["negative_filters"] = negative_filters

        print(f"Search completed in {result['search_time_ms']}ms")
        return result
