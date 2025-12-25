"""
Pydantic Models for Restaurant Search
Data validation and schema definitions
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator

class SearchRequest(BaseModel):
    """
    Request model for restaurant search
    """
    query: str = Field(..., description="User search query")
    user_lat: Optional[float] = Field(None, description="User latitude")
    user_lon: Optional[float] = Field(None, description="User longitude")
    use_cache: bool = Field(True, description="Whether to use cached results")
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v):
        """Validate query is not empty"""
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @field_validator('user_lat')
    @classmethod
    def validate_latitude(cls, v):
        """Validate latitude is in valid range"""
        if v is not None and not (-90 <= v <= 90):
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @field_validator('user_lon')
    @classmethod
    def validate_longitude(cls, v):
        """Validate longitude is in valid range"""
        if v is not None and not (-180 <= v <= 180):
            raise ValueError('Longitude must be between -180 and 180')
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "query": "best sushi near me",
                "user_lat": -43.5321,
                "user_lon": 172.6362,
                "use_cache": True
            }]
        }
    }


class Tier1Features(BaseModel):
    """
    Tier 1 enhanced features for a restaurant
    """
    hours_category: Optional[str] = None
    hours_dict: Optional[Dict[str, str]] = None
    hours_pretty: Optional[str] = None
    has_delivery: Optional[bool] = None
    has_takeout: Optional[bool] = None
    has_dine_in: Optional[bool] = None
    price_level: Optional[int] = None
    price_bucket: Optional[str] = None
    types: Optional[List[str]] = None
    avg_rating: Optional[float] = None
    num_reviews: Optional[int] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "hours_category": "lunch_dinner",
                "hours_dict": {
                    "monday": "11:30 AM-2 PM, 5-9 PM",
                    "tuesday": "11:30 AM-2 PM, 5-9 PM"
                },
                "hours_pretty": "Mon-Fri: 11:30 AM-2 PM, 5-9 PM",
                "has_delivery": True,
                "has_takeout": True,
                "has_dine_in": True,
                "price_level": 2,
                "price_bucket": "Mid-range",
                "types": ["chinese", "restaurant"],
                "avg_rating": 4.5,
                "num_reviews": 156,
                "address": "123 Main Street, Christchurch",
                "phone": "+64 3 123 4567"
            }]
        }
    }


class RestaurantResult(BaseModel):
    """
    Single restaurant search result
    """
    # ✅ FIXED: Use 'id' with alias for '_id'
    id: str = Field(..., alias="_id", description="Document ID")
    score: float = Field(..., description="Search relevance score")
    restaurant: str = Field(..., description="Restaurant name")
    restaurant_id: Optional[str] = Field(None, description="Restaurant ID")
    cuisines: List[str] = Field(default_factory=list, description="Cuisine types")
    full_review: str = Field(..., description="Full review text or summary")
    chunk_text: str = Field(..., description="Preview text")
    type: str = Field(..., description="Document type (review or restaurant_summary)")
    
    # Optional fields for reviews
    rating: Optional[float] = None
    platform: Optional[str] = None
    user: Optional[str] = None
    
    # Optional fields for summaries
    avg_rating: Optional[float] = None
    num_reviews: Optional[int] = None
    
    # Optional Tier 1 features
    tier1_features: Optional[Tier1Features] = None
    
    # Optional location data
    distance_km: Optional[float] = Field(None, description="Distance from user in km")
    distance_category: Optional[str] = Field(None, description="Human-readable distance")
    
    # Optional cross-encoder score
    cross_encoder_score: Optional[float] = None
    
    model_config = {
        "populate_by_name": True,  # Allow using both 'id' and '_id'
        "json_schema_extra": {
            "examples": [{
                "_id": "doc_123",
                "score": 0.95,
                "restaurant": "Madam Kwong",
                "restaurant_id": "rest_456",
                "cuisines": ["Chinese", "Dim Sum"],
                "full_review": "Excellent dim sum and friendly service...",
                "chunk_text": "Excellent dim sum and friendly service...",
                "type": "review",
                "rating": 5.0,
                "platform": "Google",
                "distance_km": 1.2,
                "distance_category": "Walking distance (< 1km)"
            }]
        }
    }


class LocationIntent(BaseModel):
    """
    Detected location intent from query
    """
    is_location_query: bool = Field(..., description="Whether query has location intent")
    needs_user_location: bool = Field(..., description="Whether user location is required")
    radius_km: float = Field(..., description="Search radius in kilometers")
    location_keywords: List[str] = Field(default_factory=list, description="Detected location keywords")
    urgency: str = Field(..., description="Urgency level (normal, immediate)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "is_location_query": True,
                "needs_user_location": True,
                "radius_km": 2.0,
                "location_keywords": ["nearby"],
                "urgency": "normal"
            }]
        }
    }


class QueryIntent(BaseModel):
    """
    Classified query intent
    """
    type: str = Field(..., description="Intent type (general, comparative, factual, etc.)")
    needs_reranking: bool = Field(False, description="Whether reranking is beneficial")
    is_comparative: bool = Field(False, description="Whether query is comparative")
    is_specific_dish: bool = Field(False, description="Whether query asks about specific dish")
    is_location_based: bool = Field(False, description="Whether query is location-based")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "type": "comparative",
                "needs_reranking": True,
                "is_comparative": True,
                "is_specific_dish": False,
                "is_location_based": False
            }]
        }
    }


class SearchResponse(BaseModel):
    """
    Response model for restaurant search
    """
    mode: str = Field(..., description="Search mode used")
    results: List[RestaurantResult] = Field(default_factory=list, description="Search results")
    restaurant_detected: Optional[str] = Field(None, description="Detected restaurant name")
    cuisines_detected: List[str] = Field(default_factory=list, description="Detected cuisines")
    intent: Optional[QueryIntent] = Field(None, description="Classified query intent")
    location_intent: Optional[LocationIntent] = Field(None, description="Location intent if applicable")
    user_location: Optional[Dict[str, float]] = Field(None, description="User coordinates")
    total_nearby: Optional[int] = Field(None, description="Total nearby restaurants found")
    cache_hit: bool = Field(False, description="Whether results came from cache")
    search_time_ms: Optional[float] = Field(None, description="Search time in milliseconds")
    enriched: bool = Field(False, description="Whether results include Tier 1 features")
    tier1_filters: Optional[Dict[str, Any]] = Field(None, description="Applied Tier 1 filters")
    negative_filters: Optional[Dict[str, List[str]]] = Field(None, description="Applied negative filters")
    message: Optional[str] = Field(None, description="Additional message (e.g., for location required)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "mode": "direct_filtered_cross_encoder_reranked",
                "results": [],
                "restaurant_detected": "Madam Kwong",
                "cuisines_detected": ["Chinese"],
                "cache_hit": False,
                "search_time_ms": 125.5,
                "enriched": True,
                "tier1_filters": {
                    "check_open_now": True
                }
            }]
        }
    }


class ToolCallRequest(BaseModel):
    """
    Request for tool call (from chat interface)
    """
    call_id: str = Field(..., description="Unique call ID")
    name: str = Field(..., description="Tool name")
    arguments: str = Field(..., description="JSON string of arguments")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "call_id": "call_abc123",
                "name": "smart_restaurant_search",
                "arguments": '{"query": "best pizza nearby"}'
            }]
        }
    }


class ToolCallResponse(BaseModel):
    """
    Response from tool call
    """
    type: str = Field("function_call_output", description="Response type")
    call_id: str = Field(..., description="Matching call ID")
    output: str = Field(..., description="JSON string output")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "type": "function_call_output",
                "call_id": "call_abc123",
                "output": '{"mode": "location_aware", "results": [...]}'
            }]
        }
    }


class ChatMessage(BaseModel):
    """
    Chat message for conversation history
    """
    role: str = Field(..., description="Message role (user, assistant, developer)")
    content: Any = Field(..., description="Message content (string or structured)")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        """Validate role is one of allowed values"""
        allowed_roles = ['user', 'assistant', 'developer', 'system']
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of {allowed_roles}')
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "role": "user",
                "content": "What's the best sushi restaurant?"
            }]
        }
    }


class FunctionCall(BaseModel):
    """
    Function call in chat history
    """
    type: str = Field("function_call", description="Type identifier")
    call_id: str = Field(..., description="Call ID")
    name: str = Field(..., description="Function name")
    arguments: str = Field(..., description="JSON string of arguments")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "type": "function_call",
                "call_id": "call_abc123",
                "name": "smart_restaurant_search",
                "arguments": '{"query": "best pizza"}'
            }]
        }
    }


class DisplayMessage(BaseModel):
    """
    Message for UI display
    """
    role: str = Field(..., description="Message role")
    content: Optional[str] = Field(None, description="Text content")
    name: Optional[str] = Field(None, description="Function name (for function calls)")
    arguments: Optional[str] = Field(None, description="Function arguments")
    output: Optional[str] = Field(None, description="Function output")
    
    @field_validator('role')
    @classmethod
    def validate_display_role(cls, v):
        """Validate display role"""
        allowed_roles = ['user', 'assistant', 'function_call']
        if v not in allowed_roles:
            raise ValueError(f'Display role must be one of {allowed_roles}')
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "role": "assistant",
                "content": "Here are some great sushi restaurants..."
            }]
        }
    }