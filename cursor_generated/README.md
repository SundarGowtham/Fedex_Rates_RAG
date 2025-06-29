# 🚢 Aura Shipping Intelligence Platform

A sophisticated multi-agent shipping intelligence platform with a modern Streamlit interface, providing deep, contextualized insights into shipping logistics through collaborative AI agents.

## 🏗️ Architecture

The platform uses a multi-agent framework with 5 specialized agents:

1. **Supervisor Agent** - Query deconstruction and task routing
2. **Structured Data Agent** - SQL operations and database queries (Vanna.ai + PostgreSQL)
3. **Vector Search Agent** - Semantic search and knowledge retrieval (SentenceTransformer + Qdrant)
4. **Auxiliary Intelligence Agent** - External API data gathering (weather, fuel, traffic)
5. **Synthesis & Visualization Agent** - Insights and visualizations (LLM + Plotly)

## 🛠️ Technology Stack

- **Backend**: Python 3.11+, LangGraph, FastAPI
- **UI**: Streamlit
- **Database**: PostgreSQL with PostGIS
- **Vector DB**: Qdrant
- **Cache**: Redis
- **AI/ML**: Ollama (local LLM, must be running locally), Vanna.ai, SentenceTransformers
- **Visualization**: Plotly, pydeck
- **Containerization**: Docker & Docker Compose

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- **PostgreSQL, Qdrant, and Redis running locally**
- **Ollama running locally** (see below)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Fedex_Rates_RAG
```

### 2. Environment Setup

```bash
# Copy environment template
cp config/env_template.txt .env

# Edit .env with your configuration
# (API keys, database credentials, etc.)
```

### 3. Start Required Services Locally

- **PostgreSQL**: Ensure it is running on port 5432 with the credentials in your `.env` file.
- **Qdrant**: Ensure it is running on port 6333. See [Qdrant documentation](https://qdrant.tech/documentation/) for local install instructions.
- **Redis**: Ensure it is running on port 6379.
- **Ollama**: See below.

#### Example: Check if services are running
```bash
# PostgreSQL
pg_isready -h localhost -p 5432

# Qdrant
curl http://localhost:6333/collections

# Redis
redis-cli ping
```

### 4. Start Ollama (LLM) Locally

- Download and install from [ollama.com](https://ollama.com/)
- Start Ollama:
  ```bash
  ollama serve
  # or just `ollama` if it starts the server by default
  ```
- Make sure it is accessible at `http://localhost:11434`

### 5. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 6. Run the Application

#### Option A: Using the main script
```bash
python start_aura.py
```

#### Option B: Direct Streamlit run
```bash
python run_streamlit.py
```

#### Option C: Manual Streamlit run
```bash
streamlit run src/ui/streamlit_app.py
```

### 7. Access the Application

Open your browser and navigate to: **http://localhost:8501**

## 📊 Adding Your CSV Data

The platform includes a comprehensive **Data Management** interface in the Streamlit app for uploading CSV data.

### Required CSV Formats

#### 1. Fedex Pricing Data
**File**: `fedex_pricing.csv`
**Columns**:
- `weight` (float): Package weight in pounds
- `transportation_type` (string): Type of transportation
- `zone` (integer): Shipping zone
- `service_type` (string): Service type (ground, express, priority, overnight)
- `price` (float): Shipping price in USD

**Example**:
```csv
weight,transportation_type,zone,service_type,price
1.0,ground,1,ground,8.50
2.0,ground,1,ground,9.25
5.0,ground,2,ground,12.75
```

#### 2. Zone Distance Mapping
**File**: `zone_distance_mapping.csv`
**Columns**:
- `zone` (integer): Shipping zone number
- `min_distance` (float): Minimum distance in miles
- `max_distance` (float): Maximum distance in miles

**Example**:
```csv
zone,min_distance,max_distance
1,0,150
2,151,300
3,301,600
4,601,1000
5,1001,1400
```

### Uploading Data via Streamlit

1. **Start the application** (see Quick Start section)
2. **Navigate to "Data Management"** in the sidebar
3. **Upload your CSV files**:
   - Use the file uploader for Fedex pricing data
   - Use the file uploader for zone distance mapping
4. **Preview the data** before uploading
5. **Click "Upload to Database"** to insert the data

### Uploading Data via Script

You can also upload data programmatically:

```python
import pandas as pd
from src.core.database import get_database_manager

# Load your CSV
df = pd.read_csv('your_fedex_pricing.csv')

# Upload to database
db_manager = await get_database_manager()
# ... upload logic
```

## 🎯 Using the Platform

### 1. Query Interface

The main interface allows you to:

- **Enter natural language queries** about shipping
- **Select query types** (pricing, route optimization, weather impact, etc.)
- **Set advanced options** (origin, destination, weight, service type)
- **Execute multi-agent analysis**

### 2. Example Queries

- "What are the Fedex rates for shipping a 5-pound package from New York to Los Angeles?"
- "How does weather affect shipping times from Chicago to Miami?"
- "What's the most cost-effective service for a 10-pound package going to zone 5?"
- "Show me fuel price trends and their impact on shipping costs"

### 3. Results Display

The platform provides:

- **Real-time agent execution status**
- **Database query results** with interactive tables
- **Semantic search results** from knowledge base
- **External data analysis** (weather, fuel, traffic)
- **AI-generated insights and recommendations**
- **Interactive visualizations** (charts, maps)

## 📁 Scripts and Utilities

The project includes organized utility scripts for maintenance and debugging:

### Check Scripts (`scripts/checks/`)

Utility scripts for validating and checking system components:

```bash
# Check all database tables
uv run python scripts/checks/check_all_tables.py

# Validate data integrity
uv run python scripts/checks/check_data.py

# Check fuel data API connectivity
uv run python scripts/checks/check_fuel_data.py
```

### Miscellaneous Scripts (`scripts/misc/`)

Data population and debugging utilities:

```bash
# Populate all data (uses all APIs - may hit limits)
uv run python scripts/misc/populate_all_data.py

# Populate only weather data (API limit friendly)
uv run python scripts/misc/populate_weather_only.py

# Debug data ingestion issues
uv run python scripts/misc/debug_ingestion.py
```

**Note**: The `populate_weather_only.py` script is recommended for regular updates to avoid hitting API limits.

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/aura_db

# Vector Database
QDRANT_URL=http://localhost:6333

# Cache
REDIS_URL=redis://localhost:6379

# LLM (Ollama must be running locally)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# External APIs
OPENWEATHER_API_KEY=your_key_here
FUEL_API_KEY=your_key_here
TRAFFIC_API_KEY=your_key_here
```

### Agent Configuration

Each agent can be configured in `src/agents/`:

- **Supervisor**: Query classification thresholds
- **Structured Data**: Vanna.ai model settings
- **Vector Search**: Embedding model and similarity thresholds
- **Auxiliary Intelligence**: API endpoints and retry settings
- **Synthesis**: LLM prompts and visualization preferences

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_core/
pytest tests/test_agents/
pytest tests/test_integration/
```

### Test Coverage

```bash
pytest --cov=src --cov-report=html
```

## 🐳 Docker Deployment

### Production Deployment

*Docker deployment is no longer supported. Please run all services locally as described above.*

## 📈 Monitoring and Logging

### Logs

- Application logs are output to the console by default.
- Check your local service logs for PostgreSQL, Qdrant, and Redis as needed.

### Metrics

The platform provides:

- **Agent execution times**
- **Query success rates**
- **Database performance metrics**
- **API response times**
- **System resource usage**

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check if PostgreSQL is running
   pg_isready -h localhost -p 5432
   
   # Check database logs (example for Homebrew)
   tail -f /opt/homebrew/var/log/postgresql@*/log
   ```

2. **Ollama Not Responding**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Pull the model if needed
   ollama pull llama3.2
   ```

3. **Streamlit App Not Loading**
   ```bash
   # Check if port 8501 is available
   lsof -i :8501
   
   # Restart the application
   python run_streamlit.py
   ```

### Performance Optimization

1. **Database Indexing**
   ```sql
   -- Add indexes for better performance
   CREATE INDEX idx_fedex_pricing_weight ON datasource_fedex_pricing(weight);
   CREATE INDEX idx_fedex_pricing_zone ON datasource_fedex_pricing(zone);
   ```

2. **Vector Search Optimization**
   ```python
   # Configure Qdrant for better performance
   collection_config = {
       "vectors": {
           "size": 768,
           "distance": "Cosine"
       }
   }
   ```

3. **Caching Strategy**
   ```python
   # Enable Redis caching for expensive operations
   CACHE_TTL = 3600  # 1 hour
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Aura Shipping Intelligence Platform** - Transforming shipping logistics with AI-powered insights. 