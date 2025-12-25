"""
Operating Hours Utilities
Parsing and validation of restaurant operating hours
"""

import re
import pandas as pd
from datetime import datetime, time as dt_time
from typing import Dict, Optional

# Import from config
from config import get_nz_now

def parse_hours_string(hours_str: str) -> dict:
    """
    Parse hours string into time objects.
    
    HANDLES MULTIPLE TIME PERIODS:
    - "11:30 AM-2 PM, 5-9 PM" → Check BOTH periods
    - "12-3 PM, 5-9 PM" → Check BOTH periods
    - "11 AM-9 PM" → Single period
    
    Args:
        hours_str: Hours string like "12-10 PM" or "11:30 AM-2 PM, 5-9 PM"
    
    Returns:
        dict: {
            'periods': [
                {'opens_at': time, 'closes_at': time},
                {'opens_at': time, 'closes_at': time}
            ],
            'is_24_hours': bool,
            'is_closed': bool
        }
    """
    if not hours_str or pd.isna(hours_str):
        return {'periods': [], 'is_24_hours': False, 'is_closed': True}
    
    hours_lower = hours_str.lower().strip()
    
    # Handle special cases
    if hours_lower in ['closed', 'unavailable', '']:
        return {'periods': [], 'is_24_hours': False, 'is_closed': True}
    
    if '24' in hours_lower and 'hour' in hours_lower:
        return {
            'periods': [{'opens_at': dt_time(0, 0), 'closes_at': dt_time(23, 59)}],
            'is_24_hours': True,
            'is_closed': False
        }
    
    # Split by comma first to handle multiple periods
    time_ranges = [s.strip() for s in hours_str.split(',')]
    
    periods = []
    
    for time_range in time_ranges:
        try:
            # Split on dash/hyphen
            parts = re.split(r'[-–—]', time_range)
            if len(parts) != 2:
                continue
            
            start_str = parts[0].strip()
            end_str = parts[1].strip()
            
            # Parse start and end times
            opens_at = _parse_single_time(start_str, end_str)
            closes_at = _parse_single_time(end_str)
            
            if opens_at and closes_at:
                periods.append({
                    'opens_at': opens_at,
                    'closes_at': closes_at
                })
        except Exception as e:
            print(f"Warning: Could not parse time range '{time_range}': {e}")
            continue
    
    if periods:
        return {
            'periods': periods,
            'is_24_hours': False,
            'is_closed': False
        }
    
    return {'periods': [], 'is_24_hours': False, 'is_closed': True}

def _parse_single_time(time_str: str, context_str: str = None) -> Optional[dt_time]:
    """
    Parse a single time string like "12 PM", "11 AM", "11:30 PM".
    
    Args:
        time_str: Time string
        context_str: Context for inferring AM/PM
    
    Returns:
        datetime.time object or None
    """
    time_str = time_str.strip()
    
    # Check if AM/PM is present
    has_am = 'am' in time_str.lower()
    has_pm = 'pm' in time_str.lower()
    
    # Extract numbers
    time_str_clean = re.sub(r'[^\d:]', '', time_str)
    
    if ':' in time_str_clean:
        # Has minutes: "11:30"
        hour_str, min_str = time_str_clean.split(':')
        hour = int(hour_str)
        minute = int(min_str)
    else:
        # Just hour: "12" or "11"
        hour = int(time_str_clean)
        minute = 0
    
    # Determine AM/PM
    if has_am:
        if hour == 12:
            hour = 0  # 12 AM = 00:00
    elif has_pm:
        if hour != 12:
            hour += 12  # Convert to 24-hour
    else:
        # No AM/PM specified - infer from context
        if context_str and ('pm' in context_str.lower()):
            # If end time has PM, start is likely PM too (e.g., "12-10 PM")
            if hour != 12:
                hour += 12
        elif hour < 12:
            # Ambiguous - default to PM for restaurant hours (common case)
            if hour >= 3 and hour <= 11:
                hour += 12  # 3-11 without AM/PM → assume PM
    
    return dt_time(hour, minute)


def is_restaurant_open_now(hours_dict: dict) -> dict:
    """
    Check if restaurant is currently open based on hours_dict.
    FULLY HANDLES MULTIPLE PERIODS + CROSS-MIDNIGHT SCENARIOS!
    
    Args:
        hours_dict: Dictionary like {'monday': '12-10 PM', 'tuesday': '11:30 AM-2 PM, 5 PM-2 AM', ...}
    
    Returns:
        dict with status info:
        {
            'is_open': bool,
            'status_message': str,
            'opens_at': str,
            'closes_at': str,
            'next_opening_day': str,
            'next_opening_time': str
        }
    """
    if not hours_dict:
        return {
            'is_open': False,
            'status_message': 'Hours not available',
            'opens_at': None,
            'closes_at': None,
            'next_opening_day': None,
            'next_opening_time': None
        }
    
    now = get_nz_now()
    current_day = now.strftime('%A').lower()
    current_time = now.time()
    
    # Get today's hours
    today_hours_str = hours_dict.get(current_day, '')
    today_hours = parse_hours_string(today_hours_str)
    
    # Check if 24 hours
    if today_hours['is_24_hours']:
        return {
            'is_open': True,
            'status_message': 'Currently OPEN (24 hours)',
            'opens_at': '12:00 AM',
            'closes_at': '11:59 PM',
            'next_opening_day': None,
            'next_opening_time': None
        }
    
    # Check if closed today
    if today_hours['is_closed']:
        next_opening = _find_next_opening(hours_dict, now)
        return {
            'is_open': False,
            'status_message': f"Currently CLOSED (closed on {current_day.capitalize()}s)",
            'opens_at': None,
            'closes_at': None,
            'next_opening_day': next_opening['day'],
            'next_opening_time': next_opening['time']
        }
    
    # Check if CURRENTLY OPEN in any period
    for period in today_hours['periods']:
        opens_at = period['opens_at']
        closes_at = period['closes_at']
        
        # Detect cross-midnight periods
        is_cross_midnight = closes_at < opens_at
        
        if is_cross_midnight:
            # Period crosses midnight (e.g., "5 PM-2 AM")
            if current_time >= opens_at or current_time <= closes_at:
                if current_time >= opens_at:
                    closes_at_str = closes_at.strftime('%I:%M %p').lstrip('0')
                    msg = f'Currently OPEN (closes at {closes_at_str} after midnight)'
                else:
                    closes_at_str = closes_at.strftime('%I:%M %p').lstrip('0')
                    msg = f'Currently OPEN (closes at {closes_at_str})'
                
                return {
                    'is_open': True,
                    'status_message': msg,
                    'opens_at': opens_at.strftime('%I:%M %p').lstrip('0'),
                    'closes_at': closes_at_str,
                    'next_opening_day': None,
                    'next_opening_time': None
                }
        else:
            # Normal period (e.g., "11:30 AM-2:30 PM")
            if opens_at <= current_time <= closes_at:
                closes_at_str = closes_at.strftime('%I:%M %p').lstrip('0')
                return {
                    'is_open': True,
                    'status_message': f'Currently OPEN (closes at {closes_at_str})',
                    'opens_at': opens_at.strftime('%I:%M %p').lstrip('0'),
                    'closes_at': closes_at_str,
                    'next_opening_day': None,
                    'next_opening_time': None
                }
    
    # Restaurant is CLOSED - Find next opening
    first_period = today_hours['periods'][0]
    last_period = today_hours['periods'][-1]
    
    # Check if we're BEFORE the first opening today
    if current_time < first_period['opens_at']:
        opens_at_str = first_period['opens_at'].strftime('%I:%M %p').lstrip('0')
        return {
            'is_open': False,
            'status_message': f'Currently CLOSED (opens today at {opens_at_str})',
            'opens_at': opens_at_str,
            'closes_at': last_period['closes_at'].strftime('%I:%M %p').lstrip('0'),
            'next_opening_day': 'today',
            'next_opening_time': opens_at_str
        }
    
    # Check if BETWEEN periods
    for period in today_hours['periods']:
        is_cross_midnight = period['closes_at'] < period['opens_at']
        
        if is_cross_midnight:
            if current_time <= period['closes_at']:
                continue
        else:
            if current_time > period['closes_at']:
                continue
        
        if current_time < period['opens_at']:
            opens_at_str = period['opens_at'].strftime('%I:%M %p').lstrip('0')
            return {
                'is_open': False,
                'status_message': f'Currently CLOSED (reopens today at {opens_at_str})',
                'opens_at': opens_at_str,
                'closes_at': period['closes_at'].strftime('%I:%M %p').lstrip('0'),
                'next_opening_day': 'today',
                'next_opening_time': opens_at_str
            }
    
    # No more periods today - Find next opening day
    next_opening = _find_next_opening(hours_dict, now)
    
    last_period_is_cross_midnight = last_period['closes_at'] < last_period['opens_at']
    
    if last_period_is_cross_midnight:
        status_msg = f'Currently CLOSED (closes at {last_period["closes_at"].strftime("%I:%M %p").lstrip("0")} after midnight)'
    else:
        status_msg = f'Currently CLOSED (closed at {last_period["closes_at"].strftime("%I:%M %p").lstrip("0")})'
    
    return {
        'is_open': False,
        'status_message': status_msg,
        'opens_at': first_period['opens_at'].strftime('%I:%M %p').lstrip('0'),
        'closes_at': last_period['closes_at'].strftime('%I:%M %p').lstrip('0'),
        'next_opening_day': next_opening['day'],
        'next_opening_time': next_opening['time']
    }


def _find_next_opening(hours_dict: dict, from_datetime: datetime) -> dict:
    """
    Find the next opening time starting from given datetime.
    HANDLES MULTIPLE PERIODS!
    
    Args:
        hours_dict: Operating hours dictionary
        from_datetime: Starting datetime
    
    Returns:
        dict with 'day' and 'time' keys
    """
    day_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    current_day_index = day_order.index(from_datetime.strftime('%A').lower())
    
    # Check next 7 days
    for i in range(1, 8):
        next_day_index = (current_day_index + i) % 7
        next_day_name = day_order[next_day_index]
        
        next_hours_str = hours_dict.get(next_day_name, '')
        next_hours = parse_hours_string(next_hours_str)
        
        if not next_hours['is_closed'] and next_hours['periods']:
            # Get first period's opening time
            first_period = next_hours['periods'][0]
            day_label = 'tomorrow' if i == 1 else next_day_name.capitalize()
            time_str = first_period['opens_at'].strftime('%I:%M %p').lstrip('0')
            
            return {'day': day_label, 'time': time_str}
    
    return {'day': None, 'time': None}


def compute_temporal_context(restaurant_data: dict) -> str:
    """
    Compute current open/closed status and format as context section.
    This makes the real-time computation visible to Phoenix evaluators.
    
    Args:
        restaurant_data: Dictionary with 'hours_dict' key
    
    Returns:
        Formatted temporal context string or empty string
    """
    hours_dict = restaurant_data.get('hours_dict')
    if not isinstance(hours_dict, dict) or not hours_dict:
        return ""
    
    # Get current status
    status = is_restaurant_open_now(hours_dict)
    
    # Get current NZ time for display
    now = get_nz_now()
    current_time_str = now.strftime('%I:%M %p %Z')
    current_day = now.strftime('%A')
    
    # Get today's hours string
    today_hours = hours_dict.get(now.strftime('%A').lower(), '')
    
    # Build temporal context section
    parts = [
        "[TEMPORAL CONTEXT]",
        f"Query Time: {current_time_str}",
        f"Current Day: {current_day}"
    ]
    
    # Add today's hours
    if today_hours and today_hours.lower() not in ['closed', 'unavailable', '']:
        parts.append(f"Today's Hours: {today_hours}")
    
    # Add current status
    parts.append(f"Status: {status['status_message']}")
    
    # Add specific open/close times if available
    if status['opens_at'] and status['closes_at']:
        parts.append(f"Opens: {status['opens_at']}")
        parts.append(f"Closes: {status['closes_at']}")
    
    # Add next opening if closed
    if not status['is_open'] and status['next_opening_day'] and status['next_opening_time']:
        parts.append(f"Next Opening: {status['next_opening_day']} at {status['next_opening_time']}")
    
    parts.append("[END TEMPORAL CONTEXT]")
    
    return "\n".join(parts)


def calculate_is_open_now(hours_dict: dict) -> bool:
    """
    Simple boolean check if restaurant is open NOW.
    HANDLES MULTIPLE PERIODS!
    
    Args:
        hours_dict: Operating hours dictionary
    
    Returns:
        True if open now, False otherwise
    """
    status = is_restaurant_open_now(hours_dict)
    return status['is_open']