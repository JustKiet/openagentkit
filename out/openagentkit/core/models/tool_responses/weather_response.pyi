from pydantic import BaseModel

class WeatherForecast(BaseModel):
    date: str | None
    max_temp: float | str | None
    min_temp: float | str | None
    chance_of_rain: float | str | None
    air_quality: str | int | None
    humidity: float | str | None
    condition: str | None

class CurrentWeather(BaseModel):
    date: str | None
    temp: float | str | None
    chance_of_rain: float | str | None
    air_quality: str | int | None
    humidity: float | str | None
    condition: str | None

class WeatherResponse(BaseModel):
    location: str | None
    current_weather: CurrentWeather | None
    forecast: list[WeatherForecast] | None
    error: str | None
