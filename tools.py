"""
Main Entry Point for Restaurant Search Tools
Provides backward compatibility with original tools.py
"""
from config import *
from typing import Optional
from core import init_qdrant, RestaurantSearchTools
from api import Tools, smart_search_schema

# ============================================
# INITIALIZATION FUNCTIONS
# ============================================

def get_search_instance():
    """
    Initialize and return search tools instance
    
    Returns:
        RestaurantSearchTools instance
    """
    client, model, bm25, doc_ids, doc_lookup, metadata = init_qdrant()
    
    search_instance = RestaurantSearchTools(
        qdrant_client=client,
        embedding_model=model,
        bm25_index=bm25,
        bm25_doc_ids=doc_ids,
        doc_lookup=doc_lookup,
        metadata=metadata
    )
    
    return search_instance


# ============================================
# BACKWARD COMPATIBILITY EXPORTS
# ============================================

# Initialize once (cached)
_search_instance = None

def _get_or_create_instance():
    """Get or create singleton search instance"""
    global _search_instance
    if _search_instance is None:
        _search_instance = get_search_instance()
    return _search_instance


# Export for backward compatibility
def smart_restaurant_search(query: str, use_cache: bool = True, 
                           user_lat: Optional[float] = None, 
                           user_lon: Optional[float] = None):
    """
    Backward compatible search function
    
    This allows old code like:
        import tools
        result = tools.smart_restaurant_search(query="best sushi")
    
    To work without changes
    """
    instance = _get_or_create_instance()
    return instance.smart_restaurant_search(
        query=query,
        use_cache=use_cache,
        user_lat=user_lat,
        user_lon=user_lon
    )


# Create Tools instance
def get_tools():
    """Get configured tools for OpenAI API"""
    instance = _get_or_create_instance()
    tools_container = Tools()
    tools_container.add_tool(instance.smart_restaurant_search, smart_search_schema)
    return tools_container


# Export tools instance
tools = get_tools()


# ============================================
# EXPORTS
# ============================================

__all__ = [
    # Main function
    'smart_restaurant_search',
    
    # Tools container
    'Tools',
    'tools',
    
    # Schema
    'smart_search_schema',
    
    # Initialization
    'init_qdrant',
    'get_search_instance',
    'get_tools',
]