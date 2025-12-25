"""
Geolocation Utilities
Distance calculations and location-based intent detection
"""

import math
import re
from typing import Dict


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
    

def detect_location_intent(query: str) -> Dict[str, any]:
    """
    Detect location-based intent in queries
    
    CRITICAL: Only returns is_location_query=True if EXPLICIT location keywords are present.
    Does NOT assume location intent just because user has coordinates enabled.
    
    Args:
        query: User search query
    
    Returns:
        Dictionary with location intent details:
        {
            'is_location_query': bool,        # True ONLY if location keywords found
            'needs_user_location': bool,      # Same as is_location_query
            'radius_km': float,               # Radius in km (only if location query)
            'location_keywords': list,        # Matched keywords
            'urgency': str                    # 'immediate', 'normal'
        }
    """
    query_lower = query.lower().strip()

    # Default: NO location intent
    location_intent = {
        'is_location_query': False,     # ← Defaults to False
        'needs_user_location': False,   # ← Defaults to False
        'radius_km': None,              # ← No radius by default
        'location_keywords': [],
        'urgency': 'normal'
    }

    # Location patterns with different radii
    location_patterns = {
        'immediate': {
            'keywords': [
                'right here',
                'here',
                'this place',
                'this block',
                'this street',
                'same street',
                'outside',
                'just here',
                'around this spot'
            ],
            'radius': 0.5,
            'urgency': 'immediate'
        },
        'walking': {
            'keywords': [
                'walking distance',
                'walkable',
                'walk to',
                'walkable distance',
                'on foot',
                'by foot',
                'easy walk',
                'short walk',
                'within walking range',
                'close enough to walk'
            ],
            'radius': 1.0,
            'urgency': 'normal'
        },
        'nearby': {
            'keywords': [
                'nearby',
                'near me',
                'near',
                'around me',
                'around here',
                'close to me',
                'close by',
                'not far',
                'pretty close',
                'local',
                'near where i am',
                'near my place',      
                'close to my place',  
            ],
            'radius': 2.0,
            'urgency': 'normal'
        },
        'area': {
            'keywords': [
                'in the area',
                'around',
                'around the area',
                'vicinity',
                'neighborhood',
                'near the city',
                'city area',
                'close to town',
                'around town',
                'near central',
                'near cbd'
            ],
            'radius': 5.0,
            'urgency': 'normal'
        },
        'driving': {
            'keywords': [
                'driving distance',
                'drive to',
                'short drive',
                'worth driving',
                'car ride',
                'by car',
                'not too far to drive',
                'within driving distance',
                'nearby suburbs'
            ],
            'radius': 10.0,
            'urgency': 'normal'
        }
    }

    # ========================================
    # STEP 1: Check for EXPLICIT location keywords
    # ========================================
    
    # Flatten keywords with their category info
    keyword_patterns = []
    for category, pattern_info in location_patterns.items():
        for keyword in pattern_info['keywords']:
            keyword_patterns.append((keyword, pattern_info))

    # Sort keywords by length (longest phrase first to avoid partial matches)
    keyword_patterns.sort(key=lambda x: len(x[0]), reverse=True)

    # Match against query
    matched_keyword = None
    matched_pattern = None
    
    for keyword, pattern_info in keyword_patterns:
        if keyword in query_lower:
            matched_keyword = keyword
            matched_pattern = pattern_info
            break

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
    # STEP 3: STRICT DECISION - Only set True if keywords OR distance found
    # ========================================
    
    if matched_keyword or explicit_distance:
        # Location keywords OR explicit distance detected
        location_intent['is_location_query'] = True
        location_intent['needs_user_location'] = True
        
        # Set radius
        if explicit_distance:
            # Explicit distance takes priority
            location_intent['radius_km'] = explicit_distance
            location_intent['location_keywords'] = [f"within {explicit_distance}km"]
        elif matched_pattern:
            # Use radius from matched keyword pattern
            location_intent['radius_km'] = matched_pattern['radius']
            location_intent['urgency'] = matched_pattern['urgency']
            location_intent['location_keywords'] = [matched_keyword]
    
    # ========================================
    # CRITICAL: If NO keywords found, return False
    # ========================================
    
    # If we reach here without setting is_location_query to True,
    # it remains False (no location intent detected)
    
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

