from fastmcp import FastMCP
import requests
try:
    from config import openweather_api
except ImportError:
    from mcp_servers.climate.config import openweather_api
# NASA POWER is a free public API — no authentication required

climate_mcp = FastMCP(name="Climate Server")

NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal"
OWM_BASE = "https://api.openweathermap.org/data/2.5"



# ── SHARED HELPER ──────────────────────────────────────────────────────────────

def _get_coords(city: str) -> dict:
    """Internal helper to resolve city → lat/lon via Nominatim."""
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": "climate-agent"}
        )
        data = response.json()
        if not data or not isinstance(data, list):
            return {"error": f"City '{city}' not found"}
        return {"lat": data[0]["lat"], "lon": data[0]["lon"]}
    except Exception as e:
        return {"error": f"Failed to geocode '{city}': {e}"}


# ── TOOL 1: get_coordinates ────────────────────────────────────────────────────

@climate_mcp.tool
async def get_coordinates(city: str) -> dict:
    """
    Get latitude and longitude for a given city name.
    Call this before get_climate, get_monthly_climate, or get_climate_normals
    whenever you have a city name instead of coordinates.
    Args:
        city: City name (e.g. 'Delhi', 'Mumbai', 'Chennai')
    """
    return _get_coords(city)


# ── TOOL 2: get_climate (daily) ────────────────────────────────────────────────

@climate_mcp.tool
async def get_climate(lat: str, lon: str, start_date: str, end_date: str) -> dict:
    """
    Fetch daily climate data for a location and date range.
    Returns temperature (avg/min/max), humidity, corrected precipitation, and wind speed.
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        start_date: Start date in YYYYMMDD format (e.g. '20260301')
        end_date: End date in YYYYMMDD format (e.g. '20260310')
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_date,
        "end": end_date,
        # PRECTOTCORR is more accurate than PRECTOT — corrected for bias
        # Added T2M_MAX, T2M_MIN for temperature range, RH2M for humidity
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M",
        "community": "AG",
        "format": "JSON"
    }

    response = requests.get(f"{NASA_POWER_BASE}/daily/point", params=params)
    data = response.json()

    try:
        p = data["properties"]["parameter"]
        return {
            "temperature_avg_2m":  p.get("T2M", {}),
            "temperature_max_2m":  p.get("T2M_MAX", {}),
            "temperature_min_2m":  p.get("T2M_MIN", {}),
            "humidity_2m":         p.get("RH2M", {}),
            "precipitation":       p.get("PRECTOTCORR", {}),
            "wind_speed_2m":       p.get("WS2M", {}),
        }
    except KeyError:
        return {"error": "Unexpected response format", "raw": str(data)[:500]}


# ── TOOL 3: get_monthly_climate ────────────────────────────────────────────────

@climate_mcp.tool
async def get_monthly_climate(lat: str, lon: str, start_year: str, end_year: str) -> dict:
    """
    Fetch monthly averaged climate data across a range of years.
    Use this instead of multiple daily calls when the user asks about:
    - Seasonal trends or year-over-year comparisons
    - Monthly averages (e.g. "Delhi temperature trend 2023-2025")
    - Monsoon or seasonal analysis
    This is far more efficient than calling get_climate 12+ times.
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        start_year: Start year as 4-digit string (e.g. '2023')
        end_year: End year as 4-digit string (e.g. '2025')
    Returns monthly averages grouped by parameter, with keys like '202301' (YYYYMM).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_year,
        "end": end_year,
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M",
        "community": "AG",
        "format": "JSON"
    }

    response = requests.get(f"{NASA_POWER_BASE}/monthly/point", params=params)
    data = response.json()

    try:
        p = data["properties"]["parameter"]
        return {
            "temperature_avg_2m":  p.get("T2M", {}),
            "temperature_max_2m":  p.get("T2M_MAX", {}),
            "temperature_min_2m":  p.get("T2M_MIN", {}),
            "humidity_2m":         p.get("RH2M", {}),
            "precipitation":       p.get("PRECTOTCORR", {}),
            "wind_speed_2m":       p.get("WS2M", {}),
        }
    except KeyError:
        return {"error": "Unexpected response format", "raw": str(data)[:500]}


# ── TOOL 4: get_climate_normals ────────────────────────────────────────────────

@climate_mcp.tool
async def get_climate_normals(lat: str, lon: str) -> dict:
    """
    Fetch 30-year climatological normals for a location — no date required.
    Use this when the user asks about 'typical', 'normal', 'average', or 'historical'
    climate for a location (e.g. "What is normal rainfall in Mumbai in July?",
    "Is this temperature unusual for Delhi in March?").
    Returns monthly normals with keys 'JAN', 'FEB', ..., 'DEC', and 'ANN' (annual avg).
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M",
        "community": "AG",
        "format": "JSON"
    }

    response = requests.get(f"{NASA_POWER_BASE}/climatology/point", params=params)
    data = response.json()

    try:
        p = data["properties"]["parameter"]
        return {
            "temperature_avg_2m":  p.get("T2M", {}),
            "temperature_max_2m":  p.get("T2M_MAX", {}),
            "temperature_min_2m":  p.get("T2M_MIN", {}),
            "humidity_2m":         p.get("RH2M", {}),
            "precipitation":       p.get("PRECTOTCORR", {}),
            "wind_speed_2m":       p.get("WS2M", {}),
        }
    except KeyError:
        return {"error": "Unexpected response format", "raw": str(data)[:500]}




# ── TOOL 5: get_current_weather ────────────────────────────────────────────────

@climate_mcp.tool
async def get_current_weather(lat: str, lon: str) -> dict:
    """
    Fetch live current weather conditions for a location — updated every 10 minutes.
    Use this for "right now", "current", "today's weather" queries.
    NASA POWER lags 1-2 days — always use this tool for present conditions.
    Returns temperature, feels-like, min/max, humidity, pressure, wind, visibility,
    weather description (e.g. "heavy rain", "clear sky"), and cloud cover.
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
    """
    

    params = {
        "lat": lat,
        "lon": lon,
        "appid": openweather_api,
        "units": "metric"   # °C, m/s
    }

    response = requests.get(f"{OWM_BASE}/weather", params=params)
    data = response.json()

    if data.get("cod") != 200:
        return {"error": data.get("message", "Unknown error"), "cod": data.get("cod")}

    main = data.get("main", {})
    wind = data.get("wind", {})
    weather_items = data.get("weather") or [{}]
    weather = weather_items[0] if weather_items else {}
    sys = data.get("sys", {})

    return {
        "description":       weather.get("description", ""),          # e.g. "light rain"
        "temperature_c":     main.get("temp"),
        "feels_like_c":      main.get("feels_like"),
        "temp_min_c":        main.get("temp_min"),
        "temp_max_c":        main.get("temp_max"),
        "humidity_pct":      main.get("humidity"),
        "pressure_hpa":      main.get("pressure"),
        "wind_speed_ms":     wind.get("speed"),
        "wind_direction_deg": wind.get("deg"),
        "visibility_m":      data.get("visibility"),
        "cloud_cover_pct":   data.get("clouds", {}).get("all"),
        "sunrise_utc":       sys.get("sunrise"),
        "sunset_utc":        sys.get("sunset"),
    }


# ── TOOL 6: get_forecast ───────────────────────────────────────────────────────

@climate_mcp.tool
async def get_forecast(lat: str, lon: str, days: int = 5) -> dict:
    """
    Fetch weather forecast for up to 5 days ahead in 3-hour steps (max 40 timestamps).
    Use this for "tomorrow", "this week", "next few days" queries.
    NASA POWER has no forecast capability — this is the only forward-looking tool.
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        days: Number of days to forecast (1-5, default 5)
    Returns a list of forecasts with timestamp, temperature, description, humidity, wind.
    """

    # OWM returns up to 40 3-hour slots — cap count based on days requested
    cnt = min(days * 8, 40)

    params = {
        "lat": lat,
        "lon": lon,
        "appid": openweather_api,
        "units": "metric",
        "cnt": cnt
    }

    response = requests.get(f"{OWM_BASE}/forecast", params=params)
    data = response.json()

    if data.get("cod") != "200":
        return {"error": data.get("message", "Unknown error"), "cod": data.get("cod")}

    forecasts = []
    for item in data.get("list", []):
        main = item.get("main", {})
        wind = item.get("wind", {})
        weather_items = item.get("weather") or [{}]
        weather = weather_items[0] if weather_items else {}
        forecasts.append({
            "datetime":        item.get("dt_txt"),           # e.g. "2026-03-12 15:00:00"
            "description":     weather.get("description"),
            "temperature_c":   main.get("temp"),
            "feels_like_c":    main.get("feels_like"),
            "humidity_pct":    main.get("humidity"),
            "wind_speed_ms":   wind.get("speed"),
            "precipitation_mm": item.get("rain", {}).get("3h", 0),
        })

    return {
        "city":      data.get("city", {}).get("name"),
        "forecasts": forecasts
    }


# ── TOOL 7: get_air_quality ────────────────────────────────────────────────────

@climate_mcp.tool
async def get_air_quality(lat: str, lon: str) -> dict:
    """
    Fetch current air quality and pollution data for a location.
    No equivalent exists in NASA POWER — this is the only air quality tool.
    Returns AQI (1=Good to 5=Very Poor) and individual pollutant concentrations:
    PM2.5, PM10, NO2, O3, SO2, CO, NH3.
    Essential for India intelligence — Delhi/Mumbai AQI is a major national topic.
    Use alongside get_current_weather for a complete environmental picture.
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
    """
    

    params = {
        "lat": lat,
        "lon": lon,
        "appid": openweather_api
    }

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/air_pollution",
        params=params
    )
    data = response.json()

    try:
        item = data["list"][0]
        aqi = item["main"]["aqi"]
        components = item["components"]

        aqi_labels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}

        return {
            "aqi":        aqi,
            "aqi_label":  aqi_labels.get(aqi, "Unknown"),
            "pm2_5":      components.get("pm2_5"),    # μg/m³
            "pm10":       components.get("pm10"),
            "no2":        components.get("no2"),
            "o3":         components.get("o3"),
            "so2":        components.get("so2"),
            "co":         components.get("co"),
            "nh3":        components.get("nh3"),
        }
    except (KeyError, IndexError):
        return {"error": "Unexpected response format", "raw": str(data)[:500]}
    


if __name__ == "__main__":
    climate_mcp.run(transport="sse", port=8001)