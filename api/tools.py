"""
OpenAI Tool Definitions and Tool Container
Handles tool schemas and function call execution
"""

import json
from typing import Dict, Any, Callable 

# ============================================
# TOOL SCHEMA DEFINITION
# ============================================

smart_search_schema = {
    "type": "function",
    "name": "smart_restaurant_search",
    "description": (
        "Enhanced hybrid BM25 + Vector search for Christchurch restaurant reviews with Tier 1 filtering. "
        "Location coordinates and search distance will be automatically provided when needed - "
        "do not include user_lat, user_lon, or max_distance_km in the call."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User query about restaurants"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

# ============================================
# TOOLS CONTAINER CLASS
# ============================================
class Tools:
    """
    Container for managing OpenAI tools and function calls
    
    Handles:
    - Tool registration
    - Function execution
    - Response formatting
    """
    
    def __init__(self):
        """Initialize empty tools container"""
        self.tools = {}
        self.functions = {}
    
    def add_tool(self, function: Callable, schema: Dict[str, Any]):
        """
        Register a tool with its schema
        
        Args:
            function: Python function to execute
            schema: OpenAI tool schema definition
        
        Example:
            >>> tools = Tools()
            >>> tools.add_tool(search_function, smart_search_schema)
        """
        self.tools[function.__name__] = schema
        self.functions[function.__name__] = function
    
    def get_tools(self):
        """
        Get all registered tool schemas
        
        Returns:
            List of tool schemas for OpenAI API
        """
        return list(self.tools.values())
    
    def function_call(self, tool_call_response):
        """
        Execute a function call from OpenAI response
        
        Args:
            tool_call_response: OpenAI tool call object with:
                - name: Function name
                - arguments: JSON string of arguments
                - call_id: Unique call identifier
        
        Returns:
            Dict with function call output:
            {
                "type": "function_call_output",
                "call_id": str,
                "output": JSON string
            }
        
        Example:
            >>> result = tools.function_call(tool_call_response)
            >>> print(result['output'])
        """
        # Parse arguments
        args = json.loads(tool_call_response.arguments)
        
        # Get function
        fn_name = tool_call_response.name
        fn = self.functions[fn_name]
        
        # Execute function
        result = fn(**args)
        
        # Format response
        return {
            "type": "function_call_output",
            "call_id": tool_call_response.call_id,
            "output": json.dumps(result),
        }
    
    def list_functions(self):
        """
        List all registered function names
        
        Returns:
            List of function names
        """
        return list(self.functions.keys())
    
    def has_function(self, name: str) -> bool:
        """
        Check if function is registered
        
        Args:
            name: Function name
        
        Returns:
            True if registered, False otherwise
        """
        return name in self.functions
    
    def get_schema(self, name: str) -> Dict[str, Any]:
        """
        Get schema for a specific tool
        
        Args:
            name: Function name
        
        Returns:
            Tool schema dictionary
        
        Raises:
            KeyError: If function not registered
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self.tools[name]


# ============================================
# HELPER FUNCTIONS
# ============================================
    
def get_search_tools(search_function: Callable) -> Tools:
    """
    Create and configure Tools instance for restaurant search
    
    Args:
        search_function: The smart_restaurant_search function to register
    
    Returns:
        Configured Tools instance
    
    Example:
        >>> from core.search import RestaurantSearchTools
        >>> search_instance = RestaurantSearchTools(...)
        >>> tools = get_search_tools(search_instance.smart_restaurant_search)
        >>> openai_tools = tools.get_tools()
    """
    tools = Tools()
    tools.add_tool(search_function, smart_search_schema)
    return tools


def create_tool_call_response(call_id: str, output: Any) -> Dict[str, str]:
    """
    Create formatted tool call response
    
    Args:
        call_id: Tool call identifier
        output: Function output (will be JSON serialized)
    
    Returns:
        Formatted response dictionary
    """
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(output) if not isinstance(output, str) else output,
    }


def parse_tool_arguments(arguments: str) -> Dict[str, Any]:
    """
    Safely parse tool call arguments
    
    Args:
        arguments: JSON string of arguments
    
    Returns:
        Parsed arguments dictionary
    
    Raises:
        json.JSONDecodeError: If arguments invalid JSON
    """
    return json.loads(arguments)



