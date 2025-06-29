# Miscellaneous Scripts

This folder contains various utility scripts for data population, debugging, and maintenance tasks.

## Scripts

- **`populate_all_data.py`** - Master script to populate all data tables (weather, fuel, traffic) with real API data
- **`populate_weather_only.py`** - Populates only weather data to avoid hitting API limits
- **`debug_ingestion.py`** - Debugging script for data ingestion processes

## Usage

Run these scripts from the project root directory:

```bash
# Populate all data (uses all APIs)
uv run python scripts/misc/populate_all_data.py

# Populate only weather data (API limit friendly)
uv run python scripts/misc/populate_weather_only.py

# Debug data ingestion
uv run python scripts/misc/debug_ingestion.py
```

## API Usage Notes

- **`populate_all_data.py`** - Uses all external APIs (weather, fuel, traffic) - may hit daily limits
- **`populate_weather_only.py`** - Uses only weather API - recommended for regular updates
- **`debug_ingestion.py`** - No external API calls - safe for debugging

## When to Use

- **After initial setup** - Run `populate_weather_only.py` to get weather data
- **For debugging** - Use `debug_ingestion.py` to troubleshoot data issues
- **For full data refresh** - Use `populate_all_data.py` (be mindful of API limits) 