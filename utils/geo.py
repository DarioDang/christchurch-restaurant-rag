"""
Geolocation Utilities
Distance calculations and location-based intent detection
"""

import math
import re
from typing import Dict, Optional


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula
    
    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine Formula 
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r


def get_distance_category(distance_km: float) -> str:
    """
    Categorize distance for user-friendly display
    
    Args:
        distance_km: Distance in kilometers
    
    Returns:
        Human-readable distance category
    """
    if distance_km < 0.5:
        return "Very close (< 0.5km)"
    elif distance_km < 1.0: 
        return "Walking distance (< 1km)"
    elif distance_km < 2.0:
        return "Short drive (< 2km)"
    elif distance_km < 5.0:
        return "Nearby (< 5km)"
    else:
        return "Further away (> 5km)"
    

def detect_location_intent(query: str, user_max_distance_km: Optional[float] = None) -> Dict[str, any]:
    """
    Detect location-based intent in queries.
    
    SIMPLIFIED LOGIC:
    - Keywords ONLY trigger location mode (don't set distance)
    - Distance comes from user's slider setting
    - If no slider value, use sensible default (5km)
    
    Args:
        query: User search query
        user_max_distance_km: User's slider setting (takes priority)
    
    Returns:
        Dictionary with location intent details
    """
    query_lower = query.lower().strip()

    # Default: NO location intent
    location_intent = {
        'is_location_query': False,
        'needs_user_location': False,
        'radius_km': None,
        'location_keywords': [],
        'urgency': 'normal',
        'distance_source': None
    }

    # ========================================
    # STEP 1: Simple location keyword detection
    # ========================================
    
    # Simple list of location trigger words
    location_keywords = [
        'nearby', 'near me', 'near', 'close', 'close by', 'close to me',
        'around me', 'around here', 'around', 'local', 'in the area',
        'walking distance', 'walking', 'walkable',
        'driving distance', 'drive', 'driving',
        'vicinity', 'neighborhood',
        'here', 'this area', 'my area', 'my location',
        'near my place', 'close to my place',
        'within', 'from here', 'from me'
    ]
    
    matched_keywords = [kw for kw in location_keywords if kw in query_lower]

    # ========================================
    # STEP 2: Check for explicit distance (e.g., "within 3km")
    # ========================================
    
    distance_patterns = [
        r'within (\d+(?:\.\d+)?)\s*(?:km|kilometers?)',
        r'(\d+(?:\.\d+)?)\s*(?:km|kilometers?) away',
        r'less than (\d+(?:\.\d+)?)\s*(?:km|kilometers?)',
        r'under (\d+(?:\.\d+)?)\s*(?:km|kilometers?)'
    ]

    explicit_distance = None
    for pattern in distance_patterns:
        match = re.search(pattern, query_lower)
        if match:
            explicit_distance = float(match.group(1))
            break

    # ========================================
    # STEP 3: DECISION LOGIC (Slider-First!)
    # ========================================
    
    if matched_keywords or explicit_distance:
        # Location keywords or explicit distance detected
        location_intent['is_location_query'] = True
        location_intent['needs_user_location'] = True
        location_intent['location_keywords'] = matched_keywords
        
        # PRIORITY ORDER:
        # 1. User's slider (HIGHEST PRIORITY)
        # 2. Explicit distance in query ("within 3km")
        # 3. Default fallback (5km)
        
        if user_max_distance_km is not None:
            # ✅ User set slider - USE IT!
            location_intent['radius_km'] = user_max_distance_km
            location_intent['distance_source'] = 'user_slider'
            print(f"🎯 Using slider distance: {user_max_distance_km}km")
        
        elif explicit_distance:
            # Query has "within Xkm"
            location_intent['radius_km'] = explicit_distance
            location_intent['distance_source'] = 'query_explicit'
            print(f"📏 Using query distance: {explicit_distance}km")
        
        else:
            # No slider, no explicit distance → default
            location_intent['radius_km'] = 5.0
            location_intent['distance_source'] = 'default'
            print(f"📍 Using default distance: 5km")
    
    return location_intent


def calculate_bounding_box(lat: float, lon: float, radius_km: float) -> Dict[str, float]:
    """
    Calculate bounding box for geo filtering
    
    Args:
        lat: Center latitude
        lon: Center longitude
        radius_km: Radius in kilometers
    
    Returns:
        Dictionary with min/max lat/lon
    """
    lat_delta = radius_km / 111.0  # ~111 km per degree latitude
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    
    return {
        'lat_min': lat - lat_delta,
        'lat_max': lat + lat_delta,
        'lon_min': lon - lon_delta,
        'lon_max': lon + lon_delta
    }

