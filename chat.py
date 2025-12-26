"""
Streamlit Chat Interface for Restaurant Search Agent
Refactored for Clean Architecture
"""

# ============================================
# IMPORTS - Organized by Category
# ============================================

# Standard library
import json
import random 
from typing import Dict, List, Optional, Any
import time

# Third-party
import streamlit as st
from openai import OpenAI
import streamlit_geolocation as st_geolocation

# Local modules - Using new refactored structure
from config import setup_streamlit_logging, OPENAI_CHAT_MODEL
from prompt import DEVELOPER_PROMPT, EXAMPLE_QUERIES 
from tracing import tracer
from opentelemetry.trace import Status, StatusCode
import tools


# ============================================
# CONFIGURATION & INITIALIZATION
# ============================================

# Initialize OpenAI client
client = OpenAI()

# Setup logging (optional - remove if you don't need it)
# setup_streamlit_logging()


@st.cache_resource
def init_search_system():
    """
    Initialize Qdrant search system (cached across sessions)
    
    Returns:
        RestaurantSearchTools: Initialized search instance
    """
    qdrant_client, model, bm25, doc_ids, doc_lookup, metadata = tools.init_qdrant()
    
    search_tool_instance = tools.RestaurantSearchTools(
        qdrant_client=qdrant_client,
        embedding_model=model,
        bm25_index=bm25,
        bm25_doc_ids=doc_ids,
        doc_lookup=doc_lookup,
        metadata=metadata
    )
    
    return search_tool_instance


def init_chat_tools(search_instance):
    """
    Initialize and register tools for chat
    
    Args:
        search_instance: RestaurantSearchTools instance
    
    Returns:
        tools.Tools: Configured tools container
    """
    chat_tools = tools.Tools()
    
    try:
        # Wrap search with Phoenix tracing
        decorated_search = tracer.tool(
            name="smart_restaurant_search",
            description="Hybrid BM25 + Vector search for restaurant reviews with Tier 1 filtering"
        )(search_instance.smart_restaurant_search)
        
        chat_tools.add_tool(decorated_search, tools.smart_search_schema)
        
    except Exception as e:
        st.error(f"Tool registration failed: {e}")
        st.stop()
    
    return chat_tools


# ============================================
# SESSION STATE MANAGEMENT
# ============================================

def init_session_state():
    """Initialize all session state variables"""
    
    # Chat history (for OpenAI API)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "developer", "content": DEVELOPER_PROMPT}
        ]
    
    # Display messages (for UI only)
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    
    # UI state
    if "pending_response" not in st.session_state:
        st.session_state.pending_response = False
    
    if "show_examples" not in st.session_state:
        st.session_state.show_examples = True
    
    if "has_user_interacted" not in st.session_state:
        st.session_state.has_user_interacted = False
    
    # Location state
    if 'location_toggle_enabled' not in st.session_state:
        st.session_state.location_toggle_enabled = False
    
    if 'location_enabled' not in st.session_state:
        st.session_state.location_enabled = False


def cleanup_chat_history(max_messages: int = 25):
    """
    Keep chat history manageable
    
    Args:
        max_messages: Maximum number of messages to keep
    """
    if len(st.session_state.chat_messages) > max_messages:
        # Keep first message (system prompt) + last N messages
        keep_count = max_messages - 1
        st.session_state.chat_messages = [
            st.session_state.chat_messages[0]  # System prompt
        ] + st.session_state.chat_messages[-keep_count:]


# ============================================
# LOCATION HANDLING
# ============================================

def get_user_coordinates() -> tuple[Optional[float], Optional[float], bool]:
    """
    Get user coordinates from session state
    
    Returns:
        tuple: (lat, lon, is_valid)
    """
    user_lat = st.session_state.get("user_lat")
    user_lon = st.session_state.get("user_lon")
    location_enabled = st.session_state.get("location_enabled", False)
    
    # Try to convert to float and validate
    try:
        if user_lat is not None and user_lon is not None and location_enabled:
            lat_float = float(user_lat)
            lon_float = float(user_lon)
            
            # Validate NZ coordinate range
            if -50 <= lat_float <= -30 and 160 <= lon_float <= 180:
                return lat_float, lon_float, True
    except (ValueError, TypeError):
        pass
    
    return None, None, False


def inject_location_to_tool_args(tool_args: Dict[str, Any], span=None) -> Dict[str, Any]:
    """
    Inject user location into tool arguments if available
    Also logs to Phoenix for tracking
    
    Args:
        tool_args: Original tool arguments
        span: Optional Phoenix span to log coordinates to  # ← ADD THIS!
    
    Returns:
        Updated tool arguments with location
    """
    lat, lon, is_valid = get_user_coordinates()
    
    if is_valid:
        tool_args["user_lat"] = lat
        tool_args["user_lon"] = lon
        print(f"🗺️ INJECTING LOCATION: lat={lat:.6f}, lon={lon:.6f}")

    else:
        tool_args["user_lat"] = None
        tool_args["user_lon"] = None
        
        print(f"❌ LOCATION NOT AVAILABLE")
    
    return tool_args


# ============================================
# UI RENDERING FUNCTIONS
# ============================================

def render_page_header(search_instance):
    """Render page title and stats - Compact version"""
    st.markdown(
        """
        <h1 style="text-align: center; margin-bottom: 0.2em; font-size: 2em;">
            🍽️ Christchurch Restaurants Review Assistant
        </h1>
        <p style="text-align: center; margin-top: 0; margin-bottom: 0.5em; color: #9ca3af; font-size: 1em;">
            Your AI-powered guide to the best restaurants in Christchurch
        </p>
        """,
        unsafe_allow_html=True
    )
    
    # Compact stats section
    left, col1, col2, col3, right = st.columns([1, 2, 2, 2, 1])
    
    with col1:
        st.markdown("""
            <div style='text-align: center; padding: 0.5em 0;'>
                <div style='font-size: 1.5em;'>🏪</div>
                <div style='font-size: 1.3em; font-weight: 600; margin: 0.2em 0;'>{}</div>
                <div style='color: #9ca3af; font-size: 0.85em;'>Restaurants</div>
            </div>
        """.format(len(search_instance.all_restaurants)), unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 0.5em 0;'>
                <div style='font-size: 1.5em;'>🍽️</div>
                <div style='font-size: 1.3em; font-weight: 600; margin: 0.2em 0;'>{}</div>
                <div style='color: #9ca3af; font-size: 0.85em;'>Cuisines</div>
            </div>
        """.format(len(search_instance.all_cuisines)), unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='text-align: center; padding: 0.5em 0;'>
                <div style='font-size: 1.5em;'>⭐</div>
                <div style='font-size: 1.3em; font-weight: 600; margin: 0.2em 0;'>{:,}</div>
                <div style='color: #9ca3af; font-size: 0.85em;'>Reviews</div>
            </div>
        """.format(search_instance.metadata['total_docs']), unsafe_allow_html=True)
    
    st.divider()

def render_welcome_section():
    """Display welcome card with example queries - only on first load with animations"""
    
    # ═══════════════════════════════════════
    # CSS ANIMATIONS
    # ═══════════════════════════════════════
    st.markdown("""
        <style>
        /* Fade-in animation for welcome card */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Fade-in animation for buttons */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Apply animation to welcome card */
        .welcome-card {
            animation: fadeInDown 0.6s ease-out;
        }
        
        /* Apply animation to assistant message (Quick Start) */
        .stChatMessage {
            animation: fadeInUp 0.8s ease-out 0.3s backwards;
        }
        
        /* Button styling with hover effects */
        .example-section .stButton > button {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            border-radius: 8px !important;
            padding: 0.75em 1em !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
            animation: fadeInUp 0.6s ease-out backwards;
        }
        
        /* Staggered animation for each button */
        .example-section .stButton:nth-child(1) > button {
            animation-delay: 0.4s;
        }
        .example-section .stButton:nth-child(2) > button {
            animation-delay: 0.5s;
        }
        .example-section .stButton:nth-child(3) > button {
            animation-delay: 0.6s;
        }
        .example-section .stButton:nth-child(4) > button {
            animation-delay: 0.7s;
        }
        
        .example-section .stButton > button:hover {
            background-color: rgba(102, 126, 234, 0.1) !important;
            border-color: rgba(102, 126, 234, 0.3) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════
    # BLOCK 1: Welcome Card (Standalone) with Animation
    # ═══════════════════════════════════════
    st.markdown("""
        <div class='welcome-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2em; 
                    border-radius: 12px;  
                    margin: 1em 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <h2 style='color: white; margin: 0; font-size: 1.8em; font-weight: 600;'>
                👋 Welcome!
            </h2>
            <p style='color: white; margin-top: 0.8em; margin-bottom: 0; font-size: 1.1em; line-height: 1.6;'>
                I can help you find restaurants, reviews, check hours, and more. 
                Ask me anything about Christchurch dining!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════
    # BLOCK 2: Example Query Buttons (Random 4 from 8)
    # ═══════════════════════════════════════
    
    # Store selected queries in session state to prevent re-randomization
    if 'selected_example_queries' not in st.session_state:
        st.session_state.selected_example_queries = random.sample(EXAMPLE_QUERIES, 4)
    
    selected_queries = st.session_state.selected_example_queries
    
    # Container with assistant message styling
    with st.chat_message("assistant"):
        st.markdown("💡 **Quick Start** - Try one of these:")
        
        # Wrap in a div for CSS targeting
        st.markdown('<div class="example-section">', unsafe_allow_html=True)
        
        cols = st.columns(2)
        for i, example in enumerate(selected_queries):
            with cols[i % 2]:
                if st.button(
                    f"🔍 {example}",
                    key=f"welcome_example_{i}_{hash(example)}",
                    use_container_width=True
                ):
                    # Hide welcome section
                    st.session_state.show_examples = False
                    st.session_state.has_user_interacted = True
                    
                    # Clear the stored queries so new ones appear on next visit
                    if 'selected_example_queries' in st.session_state:
                        del st.session_state.selected_example_queries
                    
                    # Add to chat
                    st.session_state.chat_messages.append(
                        {"role": "user", "content": example}
                    )
                    st.session_state.display_messages.append(
                        {"role": "user", "content": example}
                    )
                    
                    st.session_state.pending_response = True
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)



def render_popular_now():
    """Show trending searches or featured restaurants"""
    st.markdown("### 🔥 Popular Right Now")
    
    popular = [
        ("🍕 Best Pizza", "Best Italian pizza recommendations"),
        ("🍔 Delivery", "Restaurants offering delivery services"),
        ("🌮 Late Night", "Restaurants open after 10 PM"),
    ]
    
    for emoji_title, description in popular:
        if st.button(
            emoji_title,
            key=f"popular_{emoji_title.replace(' ', '_')}",
            use_container_width=True,
            help=description
        ):
            # Extract query from title (remove emoji)
            query = emoji_title.split(" ", 1)[1]  # e.g., "Best Pizza" → "Best Pizza"
            
            # Add to chat
            st.session_state.show_examples = False
            st.session_state.has_user_interacted = True
            st.session_state.chat_messages.append({"role": "user", "content": query})
            st.session_state.display_messages.append({"role": "user", "content": query})
            st.session_state.pending_response = True
            st.rerun()

def render_chat_message(msg: Dict[str, str]):
    """
    Render a single chat message
    
    Args:
        msg: Message dict with role and content
    """
    role = msg["role"]
    
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    
    elif role == "function_call":
        with st.chat_message("assistant"):
            with st.expander(f"🔧 Tool Call: {msg['name']}"):
                st.code(f"Function: {msg['name']}", language="text")
                st.code(f"Arguments: {msg['arguments']}", language="json")
                
                # Parse and display results nicely
                try:
                    output = json.loads(msg['output'])
                    
                    # Show mode
                    if 'mode' in output:
                        st.info(f"Search Mode: **{output['mode']}**")
                    
                    # Show location info
                    if output.get('mode') == 'location_aware':
                        if output.get('location_intent'):
                            radius = output['location_intent'].get('radius_km', 'N/A')
                            st.success(f"🗺️ Location Search: {radius}km radius")
                        if output.get('total_nearby'):
                            st.info(f"📍 Found {output['total_nearby']} nearby restaurants")
                    
                    # Show Tier 1 filters
                    if output.get('tier1_filters'):
                        st.success(f"🎯 Tier 1 Filters: {output['tier1_filters']}")
                    
                    # Show detections
                    if output.get('restaurant_detected'):
                        st.success(f"🏪 Restaurant: {output['restaurant_detected']}")
                    if output.get('cuisines_detected'):
                        st.success(f"🍽️ Cuisines: {', '.join(output['cuisines_detected'])}")
                    
                    # Show result count
                    results = output.get('results', [])
                    if isinstance(results, dict) and 'results' in results:
                        results = results['results']
                    st.metric("Results Retrieved", len(results))
                    
                    # Show cache status
                    if output.get('cache_hit'):
                        st.info("⚡ Results from cache (10x faster!)")
                
                except:
                    st.code(msg['output'], language="json")


def render_sidebar(search_instance):
    """Render sidebar with location settings and user-friendly tips"""
    with st.sidebar:
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. LOCATION SETTINGS (Keep as is)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.subheader("📍 Location Settings")
        
        enable_location = st.toggle(
            "Enable Location Detection",
            value=st.session_state.location_toggle_enabled,
            help="Allow the app to use your location for nearby restaurant searches"
        )
        
        st.session_state.location_toggle_enabled = enable_location
        
        if enable_location:
            location = st_geolocation.streamlit_geolocation()
            
            if location and location.get('latitude') and location.get('longitude'):
                lat, lon = location['latitude'], location['longitude']
                
                st.session_state.user_lat = lat
                st.session_state.user_lon = lon
                st.session_state.location_enabled = True
                st.session_state.location_timestamp = st.session_state.get('location_timestamp', 0) + 1
                
                st.success(f"📍 Location detected")
                st.caption(f"Coordinates: {lat:.4f}, {lon:.4f}")
            
            elif st.session_state.get('user_lat') is not None and st.session_state.get('user_lon') is not None:
                lat, lon = st.session_state.user_lat, st.session_state.user_lon
                st.session_state.location_enabled = True
                
                st.success(f"📍 Location active")
                st.caption(f"Coordinates: {lat:.4f}, {lon:.4f}")
            
            else:
                st.warning("Please click the button above to trigger location.")
                st.caption("This may take a few seconds on first use")
                st.session_state.location_enabled = True
        
        else:
            st.session_state.location_enabled = False
            for key in ['user_lat', 'user_lon', 'location_timestamp']:
                if key in st.session_state:
                    del st.session_state[key]
        
        st.divider()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. POPULAR RIGHT NOW (NEW!)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        render_popular_now()

        st.divider()
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. POPULAR SEARCHES (Replace cuisines)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.subheader("🍽️ Popular Searches")

        # Cuisine tags with country flags
        st.markdown("**By Cuisine:**")
        cuisine_tags = [
            ("🇨🇳 Chinese", "Chinese"),
            ("🇯🇵 Japanese", "Japanese"),
            ("🇮🇹 Italian", "Italian"),
            ("🇹🇭 Thai", "Thai"),
            ("🇮🇳 Indian", "Indian"),
            ("🇻🇳 Vietnam", "Vietnam"),
            ("🇰🇷 Korean", "Korean"),
            ("🇺🇸 American", "American"),
        ]

        cols = st.columns(2)
        for i, (display_name, cuisine) in enumerate(cuisine_tags):
            with cols[i % 2]:
                if st.button(display_name, key=f"cuisine_{cuisine}", use_container_width=True):
                    # Add query to chat
                    query = f"Best {cuisine} restaurants"
                    st.session_state.show_examples = False
                    st.session_state.has_user_interacted = True
                    st.session_state.chat_messages.append({"role": "user", "content": query})
                    st.session_state.display_messages.append({"role": "user", "content": query})
                    st.session_state.pending_response = True
                    st.rerun()

        st.markdown("**By Service:**")
        service_cols = st.columns(2)
        services = [
            ("🚚 Delivery", "delivery"),
            ("🥡 Takeout", "takeout"),
            ("💰 Budget", "cheap"),
            ("🌙 Late Night", "open late")
        ]

        for i, (label, keyword) in enumerate(services):
            with service_cols[i % 2]:
                if st.button(label, key=f"service_{keyword}", use_container_width=True):
                    query = f"Restaurants with {keyword}"
                    st.session_state.show_examples = False
                    st.session_state.has_user_interacted = True
                    st.session_state.chat_messages.append({"role": "user", "content": query})
                    st.session_state.display_messages.append({"role": "user", "content": query})
                    st.session_state.pending_response = True
                    st.rerun()
        
        st.divider()
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. SYSTEM INFO (Move to bottom, collapsible)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with st.expander("ℹ️ System Info", expanded=False):
            st.caption("**Database Stats:**")
            st.caption(f"• {len(search_instance.all_restaurants)} restaurants")
            st.caption(f"• {len(search_instance.all_cuisines)} cuisine types")
            st.caption(f"• {search_instance.metadata['total_docs']:,} reviews")
            
            st.caption("\n**Search Technology:**")
            st.caption("• Hybrid BM25 + Vector search")
            st.caption("• Location-aware geo filtering")
            st.caption("• Real-time availability checking")


# ============================================
# LLM INTERACTION LOGIC
# ============================================

def process_llm_response(chat_tools: tools.Tools):
    """
    Main LLM processing loop - handles tool calls and responses
    
    Args:
        chat_tools: Configured tools container
    """
    with tracer.start_as_current_span("assistant-turn", openinference_span_kind="chain") as chain_span:
        try:
            # Get user input for tracing
            user_input = st.session_state.chat_messages[-1]["content"]
            chain_span.set_attribute("input.value", user_input)
        except:
            chain_span.set_attribute("input.value", "")
        
        try:
            # Custom multi-stage spinner
            with st.spinner("🔍 Searching restaurants..."):
                time.sleep(0.3)
            with st.spinner("📊 Analyzing reviews..."):
                time.sleep(0.2)
            with st.spinner("✨ Preparing results..."):
                while True:
                    with tracer.start_as_current_span(
                        "Responses.create",
                        openinference_span_kind="llm"
                    ) as llm_span:
                        
                        llm_span.set_attribute("llm.input", str(st.session_state.chat_messages))
                        tools_available = chat_tools.get_tools()
                        llm_span.set_attribute("llm.model_name", OPENAI_CHAT_MODEL)
                        llm_span.set_attribute("llm.tools.count", len(tools_available))
                        
                        # Cleanup history before API call
                        cleanup_chat_history(max_messages=25)
                        
                        # Call OpenAI API
                        response = client.responses.create(
                            model=OPENAI_CHAT_MODEL,
                            input=st.session_state.chat_messages,
                            tools=tools_available,
                        )
                        
                        assistant_message = ""
                        tool_called = False
                        
                        # Process response
                        for entry in response.output:
                            
                            # Handle function calls
                            if entry.type == "function_call":
                                tool_called = True
                                
                                print("=" * 50)
                                print(f"🔧 FUNCTION CALL: {entry.name}")
                                print(f"🔧 Arguments: {entry.arguments}")
                                print("=" * 50)
                                
                                # Append function call to history
                                st.session_state.chat_messages.append({
                                    "type": "function_call",
                                    "call_id": entry.call_id,
                                    "name": entry.name,
                                    "arguments": entry.arguments
                                })
                                
                                # Parse and inject location
                                tool_args = json.loads(entry.arguments)

                                # Store original lat/lon before injection
                                original_lat = tool_args.get("user_lat")
                                original_lon = tool_args.get("user_lon")

                                tool_args = inject_location_to_tool_args(tool_args, span=llm_span)

                                # Get updated coordinates
                                lat = tool_args.get("user_lat")
                                lon = tool_args.get("user_lon")
                                
                                # Execute tool
                                class UpdatedToolCall:
                                    def __init__(self, original_entry, new_arguments):
                                        self.call_id = original_entry.call_id
                                        self.name = original_entry.name
                                        self.arguments = new_arguments
                                
                                updated_entry = UpdatedToolCall(entry, json.dumps(tool_args))
                                result = chat_tools.function_call(updated_entry)

                                #  Log location attributes to LLM span 
                                if lat is not None and lon is not None:
                                    llm_span.set_attribute("user.location.enabled", True)
                                    llm_span.set_attribute("user.location.latitude", lat)
                                    llm_span.set_attribute("user.location.longitude", lon)
                                    llm_span.set_attribute("user.location.coordinates", f"{lat},{lon}")
                                    llm_span.set_attribute("user.location.city", "Christchurch")
                                    llm_span.set_attribute("user.location.country", "New Zealand")
                                    llm_span.set_attribute("user.location.lat_rounded", round(lat, 2))
                                    llm_span.set_attribute("user.location.lon_rounded", round(lon, 2))
                                    llm_span.set_attribute("user.location.timestamp", 
                                                         st.session_state.get('location_timestamp', 0))
                                    print(f"📊 LOGGED TO PHOENIX: lat={lat:.6f}, lon={lon:.6f}")
                                else:
                                    llm_span.set_attribute("user.location.enabled", False)
                                    llm_span.set_attribute("user.location.reason", "Location not shared by user")
                                
                                # Extract reference text for Phoenix QA
                                try:
                                    parsed = json.loads(result["output"])
                                    docs = parsed.get("results", [])
                                    
                                    reference_text = ""
                                    for d in docs:
                                        ref = d.get("full_review")
                                        if ref:
                                            reference_text += ref + "\n\n"
                                    
                                    chain_span.set_attribute("reference", reference_text)
                                except Exception as e:
                                    chain_span.set_attribute("reference_error", str(e))
                                
                                # Phoenix tracing
                                llm_span.set_attribute("tool.name", entry.name)
                                llm_span.set_attribute("tool.input", entry.arguments)
                                llm_span.set_attribute("tool.output", result["output"])
                                
                                # Append result to history
                                st.session_state.chat_messages.append(result)
                                
                                # Display in UI
                                st.session_state.display_messages.append({
                                    "role": "function_call",
                                    "name": entry.name,
                                    "arguments": entry.arguments,
                                    "output": result["output"],
                                })
                            
                            # Handle assistant messages
                            elif entry.type == "message":
                                try:
                                    msg = entry.content[0]
                                    assistant_message = (
                                        getattr(msg, "text", "")
                                        or getattr(msg, "refusal", "")
                                    )
                                except:
                                    assistant_message = ""
                                
                                # Append to history
                                if assistant_message:
                                    st.session_state.chat_messages.append({
                                        "role": "assistant",
                                        "content": [
                                            {
                                                "type": "output_text",
                                                "text": assistant_message
                                            }
                                        ]
                                    })
                        
                        # Display assistant message
                        if assistant_message:
                            llm_span.set_attribute("llm.output", assistant_message)
                            st.session_state.display_messages.append({
                                "role": "assistant",
                                "content": assistant_message,
                            })
                            chain_span.set_attribute("output.value", assistant_message)
                        
                        # Stop loop when no tool call
                        if not tool_called:
                            break
        
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.error(traceback.format_exc())
            chain_span.record_exception(e)
            chain_span.set_status(Status(StatusCode.ERROR))
        
        else:
            chain_span.set_status(Status(StatusCode.OK))
        
        # Reset state
        st.session_state.show_examples = False
        st.session_state.pending_response = False
        st.rerun()


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point"""
    
    # Configure page
    st.set_page_config(
        page_title="Christchurch Restaurant Review Agent",
        page_icon="🍜",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Load search system
    with st.spinner("Loading search system..."):
        search_instance = init_search_system()
        chat_tools = init_chat_tools(search_instance)
    
    # Render UI
    render_page_header(search_instance)

    # Show welcome section (only on first load)
    if st.session_state.show_examples:
        render_welcome_section()
    
    # Show chat history
    for msg in st.session_state.display_messages:
        render_chat_message(msg)
    
    # Chat input
    prompt = st.chat_input("Ask about restaurants in Christchurch...")
    
    if prompt:
        st.session_state.show_examples = False
        st.session_state.has_user_interacted = True
        
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.session_state.display_messages.append({"role": "user", "content": prompt})
        
        st.session_state.pending_response = True
        st.rerun()
    
    # Process pending LLM response
    if st.session_state.pending_response:
        process_llm_response(chat_tools)
    
    # Render sidebar
    render_sidebar(search_instance)


# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()