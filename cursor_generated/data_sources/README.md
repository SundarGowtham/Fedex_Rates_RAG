# 📁 Data Sources Directory

This directory contains CSV files that will be automatically ingested into PostgreSQL when the application starts.

## 🚀 How It Works

1. **Place your CSV files** in this `data_sources/` folder
2. **Name them correctly** (see formats below)
3. **Start the application** - data will be automatically ingested
4. **Use the Streamlit app** to manage data (clear, re-ingest, etc.)

## 📋 Supported CSV Formats

### 1. Fedex Pricing Data
**Filename Pattern:** `fedex_pricing*.csv`

**Required Columns:**
- `weight` (float) - Package weight in pounds
- `transportation_type` (string) - Type of transportation
- `zone` (string) - Shipping zone
- `service_type` (string) - Service type (ground, express, priority, overnight)
- `price` (float) - Shipping price in USD

**Example:**
```csv
weight,transportation_type,zone,service_type,price
1.0,ground,1,ground,8.50
2.0,ground,1,ground,9.25
5.0,ground,2,ground,12.75
```

### 2. Zone Distance Mapping
**Filename Pattern:** `zone_distance*.csv`

**Required Columns:**
- `zone` (string) - Zone number
- `min_distance` (float) - Minimum distance in miles
- `max_distance` (float) - Maximum distance in miles

**Example:**
```csv
zone,min_distance,max_distance
1,0,150
2,151,300
3,301,600
```

## 🔄 Data Ingestion Process

### Automatic Ingestion
- **On startup:** The application automatically checks for CSV files
- **Validation:** Each file is validated for required columns
- **Duplicate prevention:** Data is only inserted if the table is empty
- **Error handling:** Failed ingestions are logged and reported

### Manual Management
Use the **"Data Management"** page in the Streamlit app to:
- **View ingestion status** of all tables
- **Re-ingest data** from CSV files
- **Clear all data** to start fresh
- **Upload new CSV files** manually (legacy feature)

## 🛠️ Troubleshooting

### Common Issues

1. **"Missing required columns"**
   - Check that your CSV has all required columns
   - Column names must match exactly (case-sensitive)

2. **"Data already exists"**
   - The system won't overwrite existing data
   - Use "Clear All Data" button to start fresh

3. **"Unknown CSV type"**
   - Make sure your filename matches the pattern
   - Use `fedex_pricing*.csv` or `zone_distance*.csv`

### File Naming Examples
✅ **Correct:**
- `fedex_pricing.csv`
- `fedex_pricing_2024.csv`
- `zone_distance.csv`
- `zone_distance_mapping.csv`

❌ **Incorrect:**
- `pricing.csv` (missing "fedex_" prefix)
- `zones.csv` (missing "distance" in name)
- `data.csv` (no recognized pattern)

## 📊 Data Status

Check the **"Data Management"** page in the Streamlit app to see:
- Which tables have data
- Which CSV files were processed
- Ingestion success/failure status

## 🔧 Advanced Usage

### Custom Data Types
To add support for new CSV types, modify:
- `src/utils/data_ingestion.py` - Add new CSV type configuration
- Database schema - Add corresponding table

### Batch Processing
- Place multiple CSV files in the folder
- All matching files will be processed on startup
- Each file type goes to its corresponding table 