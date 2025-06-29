# Check Scripts

This folder contains utility scripts for checking and validating various aspects of the Aura Shipping Intelligence Platform.

## Scripts

- **`check_all_tables.py`** - Checks the status and row counts of all database tables
- **`check_data.py`** - Validates data integrity and format in the database
- **`check_fuel_data.py`** - Specifically checks fuel price data and API connectivity

## Usage

Run these scripts from the project root directory:

```bash
# Check all tables
uv run python scripts/checks/check_all_tables.py

# Check data integrity
uv run python scripts/checks/check_data.py

# Check fuel data
uv run python scripts/checks/check_fuel_data.py
```

These scripts are useful for:
- Debugging database issues
- Validating data ingestion
- Monitoring API connectivity
- Ensuring data quality 