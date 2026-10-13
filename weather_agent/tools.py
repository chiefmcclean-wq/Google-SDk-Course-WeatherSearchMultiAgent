"""Weather and Geocoding Tools for Google ADK Weather Agent.

This module provides tools for:
1. Converting location names to geographic coordinates (latitude/longitude).
2. Fetching real-time weather forecasts and conditions from the National Weather Service (NWS)
   with global fallback via Open-Meteo.
3. Retrieving active weather alerts and advisories.
"""

from typing import Any, Dict, List, Optional
import json
import urllib.parse
import urllib.request


def get_lat_lon(location_name: str) -> Dict[str, Any]:
    """Retrieves latitude and longitude coordinates for a given location name.

    Args:
        location_name (str): Name of the city, region, or address (e.g., 'Austin, TX',
            'Seattle, WA', 'Miami, FL', 'London, UK').

    Returns:
        Dict[str, Any]: A dictionary containing status, location details, latitude,
            and longitude. Returns an error status if the location cannot be resolved.
    """
    try:
        encoded = urllib.parse.quote(location_name.strip())
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=en&format=json"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GoogleADK-WeatherBot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = data.get("results")
        if not results:
            return {
                "status": "error",
                "message": f"Could not find coordinates for location: '{location_name}'. Please verify the spelling."
            }

        first = results[0]
        return {
            "status": "success",
            "name": first.get("name"),
            "region": first.get("admin1", ""),
            "country": first.get("country", ""),
            "latitude": round(float(first["latitude"]), 4),
            "longitude": round(float(first["longitude"]), 4),
            "formatted_name": f"{first.get('name')}, {first.get('admin1', '')} {first.get('country', '')}".strip(", ")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Geocoding request failed: {str(e)}"
        }


def get_weather_forecast(lat: float, lon: float) -> Dict[str, Any]:
    """Fetches the weather forecast and current conditions for given coordinates.

    Uses the U.S. National Weather Service (NWS) API for locations within the United States,
    and falls back to Open-Meteo for international locations or when NWS is unavailable.

    Args:
        lat (float): Latitude of the location (e.g., 30.2672).
        lon (float): Longitude of the location (e.g., -97.7431).

    Returns:
        Dict[str, Any]: A dictionary with forecast details including current temperature,
            weather summary, detailed period forecast, and winds.
    """
    # 1. Attempt National Weather Service (NWS) API (ideal for US cities)
    try:
        nws_points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        nws_headers = {
            "User-Agent": "GoogleADKWeatherBot/1.0 (adk-weather-bot@example.com)",
            "Accept": "application/geo+json"
        }
        points_req = urllib.request.Request(nws_points_url, headers=nws_headers)
        with urllib.request.urlopen(points_req, timeout=6) as resp:
            points_data = json.loads(resp.read().decode("utf-8"))

        forecast_url = points_data["properties"]["forecast"]
        forecast_req = urllib.request.Request(forecast_url, headers=nws_headers)
        with urllib.request.urlopen(forecast_req, timeout=6) as f_resp:
            f_data = json.loads(f_resp.read().decode("utf-8"))

        periods = f_data["properties"].get("periods", [])
        if periods:
            current = periods[0]
            upcoming = [
                {
                    "period": p.get("name"),
                    "temperature": f"{p.get('temperature')}°{p.get('temperatureUnit')}",
                    "short_forecast": p.get("shortForecast"),
                    "detailed_forecast": p.get("detailedForecast")
                }
                for p in periods[1:4]
            ]

            return {
                "status": "success",
                "source": "U.S. National Weather Service (NWS)",
                "current_period": current.get("name"),
                "temperature": f"{current.get('temperature')}°{current.get('temperatureUnit')}",
                "short_forecast": current.get("shortForecast"),
                "detailed_forecast": current.get("detailedForecast"),
                "wind": f"{current.get('windSpeed')} {current.get('windDirection')}",
                "relative_humidity": f"{current.get('relativeHumidity', {}).get('value', 'N/A')}%" if current.get('relativeHumidity') else "N/A",
                "upcoming_periods": upcoming
            }
    except Exception:
        # Fall through to global weather service
        pass

    # 2. Global fallback via Open-Meteo
    try:
        om_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat:.4f}&longitude={lon:.4f}&current_weather=true"
            f"&temperature_unit=fahrenheit&windspeed_unit=mph"
        )
        req = urllib.request.Request(om_url, headers={"User-Agent": "GoogleADK-WeatherBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            om_data = json.loads(resp.read().decode("utf-8"))

        curr = om_data.get("current_weather", {})
        temp_f = curr.get("temperature")
        temp_c = round((temp_f - 32) * 5 / 9, 1) if temp_f is not None else None

        # WMO weather code mapping
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        condition = weather_codes.get(curr.get("weathercode", -1), "Fair / Mixed")

        return {
            "status": "success",
            "source": "Open-Meteo Global Weather",
            "temperature": f"{temp_f}°F ({temp_c}°C)",
            "condition": condition,
            "wind": f"{curr.get('windspeed')} mph",
            "timestamp": curr.get("time")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve weather data: {str(e)}"
        }


def get_weather_alerts(lat: float, lon: float) -> Dict[str, Any]:
    """Checks for active weather hazards, watches, or warnings for a location.

    Queries the U.S. National Weather Service active alert system.

    Args:
        lat (float): Latitude of the location (e.g., 30.2672).
        lon (float): Longitude of the location (e.g., -97.7431).

    Returns:
        Dict[str, Any]: A dictionary listing active alerts or indicating no alerts.
    """
    try:
        url = f"https://api.weather.gov/alerts/active?point={lat:.4f},{lon:.4f}"
        headers = {
            "User-Agent": "GoogleADKWeatherBot/1.0 (adk-weather-bot@example.com)",
            "Accept": "application/geo+json"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        features = data.get("features", [])
        if not features:
            return {
                "status": "success",
                "alert_count": 0,
                "alerts": [],
                "message": "No active weather alerts or warnings for this area."
            }

        alerts_summary = []
        for feat in features[:3]:  # Top 3 most immediate
            props = feat.get("properties", {})
            alerts_summary.append({
                "event": props.get("event"),
                "headline": props.get("headline"),
                "severity": props.get("severity"),
                "urgency": props.get("urgency"),
                "description": props.get("description", "")[:250] + "..." if len(props.get("description", "")) > 250 else props.get("description", "")
            })

        return {
            "status": "success",
            "alert_count": len(features),
            "alerts": alerts_summary
        }
    except Exception:
        return {
            "status": "success",
            "alert_count": 0,
            "alerts": [],
            "message": "Weather alerts service unavailable or location outside US."
        }


def get_current_weather_for_location(location: str) -> Dict[str, Any]:
    """Convenience tool to get full weather information directly for a city or region name.

    Performs geocoding, retrieves forecast/conditions, and checks active weather alerts.

    Args:
        location (str): Name of the city, state, or location (e.g. 'Austin, TX', 'New York', 'Seattle').

    Returns:
        Dict[str, Any]: Complete weather report including location coordinates,
            temperature, forecast, and active alerts.
    """
    loc_res = get_lat_lon(location)
    if loc_res.get("status") != "success":
        return loc_res

    lat = loc_res["latitude"]
    lon = loc_res["longitude"]

    weather_res = get_weather_forecast(lat, lon)
    alerts_res = get_weather_alerts(lat, lon)

    return {
        "status": "success",
        "location": loc_res["formatted_name"],
        "coordinates": {"latitude": lat, "longitude": lon},
        "forecast": weather_res,
        "alerts": alerts_res
    }
