"""
Custom Evaluators for Restaurant RAG System
Includes Location, Metadata, and Shadow Geographic evaluators
"""

import re
import json
import math
from typing import Dict, Optional


class LocationAccuracyEvaluator:
    """
    Evaluates if location-based responses are accurate
    Checks distance formatting and sorting
    """
    
    def __init__(self, eval_model):
        """
        Initialize evaluator
        
        Args:
            eval_model: OpenAI model for evaluation
        """
        self.eval_model = eval_model
    
    def evaluate(self, input_text: str, output_text: str, reference_text: str) -> Dict:
        """
        Evaluate location-based response accuracy
        
        Args:
            input_text: User query
            output_text: Assistant response
            reference_text: Retrieved context
        
        Returns:
            Dict with label and explanation
        """
        # Check if this is a location query
        location_keywords = [
            'nearby', 'near me', 'close', 'walking distance', 
            'around here', 'near', 'vicinity', 'area'
        ]
        is_location_query = any(
            keyword in input_text.lower() 
            for keyword in location_keywords
        )
        
        if not is_location_query:
            return {
                "label": "not_applicable",
                "explanation": "Not a location query"
            }
        
        # Extract distance information from output
        distance_pattern = r'(\d+\.?\d*)\s*km\s+away'
        distances = re.findall(distance_pattern, output_text)
        
        if not distances:
            return {
                "label": "no_distances",
                "explanation": "Location query but no distance information provided in expected format"
            }
        
        # Check if distances are reasonable and sorted
        try:
            distance_values = [float(d) for d in distances]
            is_sorted = distance_values == sorted(distance_values)
            
            # Check for reasonable distances (within expected search radius)
            max_reasonable_distance = 15.0  # 15km for Christchurch area
            all_reasonable = all(d <= max_reasonable_distance for d in distance_values)
            
            if is_sorted and all_reasonable:
                return {
                    "label": "accurate",
                    "explanation": f"Distances properly sorted and reasonable: {distance_values}"
                }
            elif not is_sorted:
                return {
                    "label": "unsorted",
                    "explanation": f"Distances not sorted by proximity: {distance_values}"
                }
            else:
                return {
                    "label": "unreasonable",
                    "explanation": f"Some distances seem unreasonable (>15km): {distance_values}"
                }
                
        except ValueError:
            return {
                "label": "invalid_distances",
                "explanation": f"Could not parse distances as numbers: {distances}"
            }


class MetadataEnrichmentEvaluator:
    """
    Evaluates if responses include expected metadata
    Checks for address, hours, phone, services, price, rating
    """
    
    def __init__(self, eval_model):
        """
        Initialize evaluator
        
        Args:
            eval_model: OpenAI model for evaluation
        """
        self.eval_model = eval_model
    
    def evaluate(self, input_text: str, output_text: str, reference_text: str) -> Dict:
        """
        Evaluate metadata enrichment in response
        
        Args:
            input_text: User query
            output_text: Assistant response
            reference_text: Retrieved context with metadata
        
        Returns:
            Dict with label and explanation
        """
        # Check if reference contains metadata
        has_metadata = '[RESTAURANT INFO]' in reference_text
        
        if not has_metadata:
            return {
                "label": "no_metadata",
                "explanation": "No [RESTAURANT INFO] metadata in reference text"
            }
        
        # Extract metadata types from reference
        metadata_types = []
        if 'Address:' in reference_text:
            metadata_types.append('address')
        if 'Phone:' in reference_text:
            metadata_types.append('phone')
        if 'Operating Hours:' in reference_text:
            metadata_types.append('hours')
        if 'Services:' in reference_text:
            metadata_types.append('services')
        if 'Price Range:' in reference_text:
            metadata_types.append('price')
        if 'Average Rating:' in reference_text:
            metadata_types.append('rating')
        
        if not metadata_types:
            return {
                "label": "no_metadata",
                "explanation": "No parseable metadata fields found"
            }
        
        # Check if output uses this metadata appropriately
        metadata_used = []
        
        # Comprehensive checks for metadata usage in output
        if any(word in output_text.lower() for word in ['address', 'located at', 'street', 'road']):
            metadata_used.append('address')
        if any(word in output_text.lower() for word in ['phone', 'call', 'contact', 'number']):
            metadata_used.append('phone')
        if any(word in output_text.lower() for word in ['open', 'close', 'hours', 'time', 'currently']):
            metadata_used.append('hours')
        if any(word in output_text.lower() for word in ['delivery', 'takeout', 'dine', 'service']):
            metadata_used.append('services')
        if any(word in output_text.lower() for word in ['rating', 'stars', '/5', 'score']):
            metadata_used.append('rating')
        if any(word in output_text.lower() for word in ['price', 'cost', '$', 'expensive', 'cheap', 'budget']):
            metadata_used.append('price')
        
        # Calculate enrichment quality
        metadata_usage_ratio = len(metadata_used) / len(metadata_types)
        
        if metadata_usage_ratio >= 0.5:  # Used 50%+ of available metadata
            return {
                "label": "well_enriched",
                "explanation": f"Used {len(metadata_used)}/{len(metadata_types)} metadata types: {metadata_used}"
            }
        elif len(metadata_used) >= 1:
            return {
                "label": "partially_enriched",
                "explanation": f"Used {len(metadata_used)}/{len(metadata_types)} metadata types: {metadata_used}"
            }
        else:
            return {
                "label": "not_enriched",
                "explanation": f"Metadata available ({metadata_types}) but not used in response"
            }


class ShadowGeographicEvaluator:
    """
    Shadow Evaluation: Validates geographic accuracy using hidden coordinate data
    
    This evaluator uses coordinate data embedded in the reference text
    (not visible to users) to validate distance calculations
    """
    
    def __init__(self, eval_model):
        """
        Initialize evaluator
        
        Args:
            eval_model: OpenAI model for evaluation
        """
        self.eval_model = eval_model
    
    def evaluate(self, input_text: str, output_text: str, reference_text: str) -> Dict:
        """
        Evaluate geographic accuracy using shadow evaluation data
        
        Args:
            input_text: User query
            output_text: Assistant response with distance claims
            reference_text: Context with hidden coordinate data
        
        Returns:
            Dict with label and explanation
        """
        print(f"[ShadowGeo] 🔍 Evaluating query: '{input_text[:50]}...'")
        
        # Check if this is a location query
        location_keywords = [
            'nearby', 'near me', 'close', 'walking distance', 
            'sushi nearby', 'bars near'
        ]
        is_location_query = any(
            keyword in input_text.lower() 
            for keyword in location_keywords
        )
        
        if not is_location_query:
            print(f"[ShadowGeo] ❌ Not a location query")
            return {
                "label": "not_applicable",
                "explanation": "Not a location query"
            }
        
        print(f"[ShadowGeo] ✅ Location query detected")
        
        # Extract shadow evaluation data from reference text
        shadow_data = self._extract_shadow_data(reference_text)
        if not shadow_data:
            print(f"[ShadowGeo] ❌ No shadow evaluation data found in reference")
            return {
                "label": "no_shadow_data",
                "explanation": "Cannot find hidden coordinate data in reference text"
            }
        
        print(f"[ShadowGeo] ✅ Found shadow data: {len(shadow_data.get('restaurants', []))} restaurants")
        
        # Extract claimed distances from user-friendly output
        claimed_distances = self._extract_claimed_distances(output_text)
        if not claimed_distances:
            print(f"[ShadowGeo] ❌ No distance claims in output")
            return {
                "label": "no_distance_claims",
                "explanation": "No distance information found in user response"
            }
        
        print(f"[ShadowGeo] ✅ Found distance claims: {claimed_distances}")
        
        # Validate geographic accuracy
        return self._validate_geographic_accuracy(shadow_data, claimed_distances)
    
    def _extract_shadow_data(self, reference_text: str) -> Optional[Dict]:
        """Extract shadow evaluation data from reference text"""
        if '[SHADOW_EVALUATION_DATA]' not in reference_text:
            print(f"[ShadowGeo] No [SHADOW_EVALUATION_DATA] section found")
            return None
        
        try:
            # Extract shadow data section
            shadow_section = reference_text.split('[SHADOW_EVALUATION_DATA]')[1]
            shadow_section = shadow_section.split('[END_SHADOW_EVALUATION_DATA]')[0]
            
            # Parse JSON data
            shadow_data = json.loads(shadow_section.strip())
            
            print(f"[ShadowGeo] Parsed shadow data successfully")
            return shadow_data
            
        except (IndexError, json.JSONDecodeError) as e:
            print(f"[ShadowGeo] Error parsing shadow data: {e}")
            return None
    
    def _extract_claimed_distances(self, output_text: str) -> Dict[str, float]:
        """Extract claimed distances from user output"""
        distance_pattern = r'([^(]+)\s*\((\d+\.?\d*)\s*km\s+away\)'
        matches = re.findall(distance_pattern, output_text)
        
        claimed = {}
        for name, distance in matches:
            restaurant_name = name.strip()
            claimed[restaurant_name] = float(distance)
        
        print(f"[ShadowGeo] Extracted {len(claimed)} distance claims")
        return claimed
    
    def _validate_geographic_accuracy(
        self, 
        shadow_data: Dict, 
        claimed_distances: Dict[str, float]
    ) -> Dict:
        """Validate geographic accuracy using Haversine distance"""
        user_coords = shadow_data.get('user_location')
        if not user_coords:
            return {
                "label": "no_user_coords",
                "explanation": "No user coordinates in shadow data"
            }
        
        user_lat = user_coords['lat']
        user_lon = user_coords['lon']
        
        errors = []
        verified_restaurants = 0
        
        for restaurant in shadow_data.get('restaurants', []):
            name = restaurant['name']
            actual_coords = restaurant.get('actual_coords')
            claimed_distance = claimed_distances.get(name)
            
            if not actual_coords or claimed_distance is None:
                continue
            
            # Calculate actual distance using Haversine formula
            actual_distance = self._haversine_distance(
                user_lat, user_lon,
                actual_coords['lat'], actual_coords['lon']
            )
            
            # Check accuracy (±20% tolerance for real-world variations)
            error_margin = abs(actual_distance - claimed_distance) / max(actual_distance, 0.1)
            
            print(
                f"[ShadowGeo] {name}: Claimed {claimed_distance}km, "
                f"Actual {actual_distance:.2f}km, Error {error_margin:.1%}"
            )
            
            verified_restaurants += 1
            
            if error_margin > 0.20:  # More than 20% error
                errors.append({
                    'restaurant': name,
                    'claimed': claimed_distance,
                    'actual': round(actual_distance, 2),
                    'error_pct': round(error_margin * 100, 1)
                })
        
        # Generate evaluation results
        if errors:
            worst = max(errors, key=lambda x: x['error_pct'])
            print(f"[ShadowGeo] ❌ Geographic errors found: {len(errors)} errors")
            return {
                "label": "geographic_error",
                "explanation": (
                    f"Distance calculation errors. Worst: {worst['restaurant']} "
                    f"claimed {worst['claimed']}km but actually {worst['actual']}km "
                    f"({worst['error_pct']}% error)"
                ),
                "details": errors[:3]  # Show top 3 errors
            }
        else:
            print(f"[ShadowGeo] ✅ All {verified_restaurants} restaurants geographically accurate")
            return {
                "label": "geographically_accurate",
                "explanation": (
                    f"All {verified_restaurants} restaurants geographically accurate "
                    f"within 20% tolerance"
                )
            }
    
    def _haversine_distance(
        self, 
        lat1: float, 
        lon1: float, 
        lat2: float, 
        lon2: float
    ) -> float:
        """
        Calculate distance using Haversine formula
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
        
        Returns:
            Distance in kilometers
        """
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return c * 6371  # Earth radius in km