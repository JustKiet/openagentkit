from typing import List, Optional, Union
from pydantic import BaseModel

class WeatherForecast(BaseModel):
    date: Optional[str]
    max_temp: Optional[Union[float, str]]
    min_temp: Optional[Union[float, str]]
    chance_of_rain: Optional[Union[float, str]]
    air_quality: Optional[Union[str, int]]
    humidity: Optional[Union[float, str]]
    condition: Optional[str]
    
class CurrentWeather(BaseModel):
    date: Optional[str]
    temp: Optional[Union[float, str]]
    chance_of_rain: Optional[Union[float, str]]
    air_quality: Optional[Union[str, int]]
    humidity: Optional[Union[float, str]]
    condition: Optional[str]
    
class WeatherResponse(BaseModel):
    location: Optional[str]
    current_weather: Optional[CurrentWeather]
    forecast: Optional[List[WeatherForecast]]
    error: Optional[str] = None