from typing import Annotated, Union, Literal, Optional
from openagentkit.core.tools.base_tool import tool
import os
import requests
from openagentkit.core.models.tool_responses import WeatherForecast, CurrentWeather, WeatherResponse
import logging
from functools import lru_cache
import time

# Set up logging
logger = logging.getLogger(__name__)

@lru_cache(maxsize=100)
def get_weather_data(location: str, days: int, api_key: str) -> dict:
    """
    Fetch weather data with caching to avoid repeated API calls.
    
    Args:
        location: Location to get weather for
        days: Number of days for forecast
        api_key: Weather API key
        
    Returns:
        Weather data dictionary or None if request failed
    """
    try:
        response = requests.get(
            url=f"https://api.weatherapi.com/v1/forecast.json",
            params={
                "key": api_key,
                "q": location,
                "days": days,
                "aqi": "yes",
                "alerts": "yes"
            },
            timeout=10  # Add timeout to prevent hanging
        )
        
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed: {str(e)}")
        return None

def get_location_from_ip() -> str:
    """
    Get user location from IP address.
    
    Returns:
        City name or "Unknown" if detection fails
    """
    try:
        response = requests.get('https://ipinfo.io', timeout=5)
        response.raise_for_status()
        data = response.json()
        city = data.get('city')
        if not city:
            logger.warning("City not found in IP info response")
            return "Unknown"
        return city
    except requests.exceptions.RequestException as e:
        logger.error(f"IP location detection failed: {str(e)}")
        return "Unknown"

def extract_current_weather(weather_data: dict) -> Optional[CurrentWeather]:
    """
    Extract current weather data from API response.
    
    Args:
        weather_data: Weather API response data
        
    Returns:
        CurrentWeather object or None if data is invalid
    """
    try:
        current_data = weather_data.get("current", {})
        if not current_data:
            return None
            
        # Get chance of rain from forecast if available
        chance_of_rain = None
        try:
            chance_of_rain = weather_data.get("forecast", {}).get("forecastday", [{}])[0].get("day", {}).get("daily_chance_of_rain")
        except (IndexError, KeyError):
            pass
            
        # Get air quality with fallback
        air_quality = "Unknown"
        try:
            air_quality = current_data.get("air_quality", {}).get("us-epa-index", "Unknown")
        except (AttributeError, KeyError):
            pass
            
        # Get condition text with fallback
        condition = "Unknown"
        try:
            condition = current_data.get("condition", {}).get("text", "Unknown")
        except (AttributeError, KeyError):
            pass
            
        return CurrentWeather(
            date=current_data.get("last_updated", "Unknown"),
            temp=current_data.get("temp_c", 0),
            chance_of_rain=chance_of_rain,
            air_quality=air_quality,
            humidity=current_data.get("humidity", 0),
            condition=condition
        )
    except Exception as e:
        logger.error(f"Error extracting current weather: {str(e)}")
        return None

def extract_forecast(weather_data: dict) -> list:
    """
    Extract forecast data from API response.
    
    Args:
        weather_data: Weather API response data
        
    Returns:
        List of WeatherForecast objects
    """
    forecasts_response = []
    
    try:
        forecasts = weather_data.get("forecast", {}).get("forecastday", [])
        
        # Skip first day (it's the current day)
        for forecast in forecasts[1:]:
            try:
                day_data = forecast.get("day", {})
                
                # Get condition with fallback
                condition = "Unknown"
                try:
                    condition = day_data.get("condition", {}).get("text", "Unknown")
                except (AttributeError, KeyError):
                    pass
                
                forecast_details = WeatherForecast(
                    date=forecast.get("date", "Unknown"),
                    max_temp=day_data.get("maxtemp_c", 0),
                    min_temp=day_data.get("mintemp_c", 0),
                    chance_of_rain=day_data.get("daily_chance_of_rain", 0),
                    air_quality=day_data.get("air_quality", "Unknown"),
                    humidity=day_data.get("avghumidity", 0),
                    condition=condition,
                )
                forecasts_response.append(forecast_details)
            except Exception as e:
                logger.error(f"Error processing forecast day: {str(e)}")
                # Continue processing other days even if one fails
                continue
    except Exception as e:
        logger.error(f"Error extracting forecast data: {str(e)}")
    
    return forecasts_response

@tool
def get_weather_tool(
    mode: Annotated[Literal["current", "forecast", "both"], "Weather Response mode."],
    location: Annotated[Union[str, Literal["Unknown"]], "The location to get the weather forecast for."], 
    days: Annotated[Union[int, Literal[0]], "The number of days to get the weather forecast for. Maximum of 3"],
    ) -> WeatherResponse:
    """
    Get weather information for a location.
    
    Args:
        mode: Type of weather data to return (current, forecast, or both)
        location: Location to get weather for, or "Unknown" to auto-detect
        days: Number of days for forecast (max 3)
        
    Returns:
        WeatherResponse object with requested weather data
    """
    # Input validation
    if days < 0:
        days = 0
    elif days > 3:
        days = 3
        
    # Get API key with proper error handling
    weather_api_key = os.getenv("WEATHERAPI_API_KEY")
    if not weather_api_key:
        logger.error("Weather API key not found in environment variables")
        return WeatherResponse(
            location="Weather API key not configured. Service unavailable.",
            current_weather=None,
            forecast=None,
            error="Weather API key not configured"
        )
    
    # Add one more day because the first forecast is for the current day
    api_days = days + 1
    
    # Determine location if not provided
    actual_location = location
    if location == "Unknown":
        actual_location = get_location_from_ip()
        
    # Retry mechanism for API calls
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Get weather data
            weather_data = get_weather_data(actual_location, api_days, weather_api_key)
            
            if not weather_data:
                if attempt < max_retries - 1:
                    logger.warning(f"Weather API request failed, retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    return WeatherResponse(
                        location=actual_location,
                        current_weather=None,
                        forecast=None,
                        error="Failed to fetch weather data after multiple attempts"
                    )
            
            # Process current weather if requested
            current_weather = None
            if mode in ["both", "current"]:
                current_weather = extract_current_weather(weather_data)
            
            # Process forecast if requested
            forecast = None
            if mode in ["both", "forecast"]:
                forecast = extract_forecast(weather_data)
            
            # Return the weather response
            return WeatherResponse(
                location=actual_location,
                current_weather=current_weather,
                forecast=forecast
            )
            
        except Exception as e:
            logger.error(f"Error in get_weather_tool: {str(e)}")
            if attempt < max_retries - 1:
                logger.warning(f"Retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Return a response with error information after all retries fail
                return WeatherResponse(
                    location=actual_location,
                    current_weather=None,
                    forecast=None,
                    error=f"Weather service error: {str(e)}"
                )