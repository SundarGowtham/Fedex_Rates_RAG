-- table_descriptions_inserts.sql
INSERT INTO table_descriptions (table_name, column_name, description) VALUES
  -- datasource_fedex_pricing
  ('datasource_fedex_pricing', 'id', 'Primary key, unique row identifier'),
  ('datasource_fedex_pricing', 'weight', 'Package weight in pounds'),
  ('datasource_fedex_pricing', 'transportation_type', 'Type of transportation (e.g., ground, air)'),
  ('datasource_fedex_pricing', 'zone', 'FedEx shipping zone code'),
  ('datasource_fedex_pricing', 'service_type', 'FedEx service type (e.g., Standard, Express)'),
  ('datasource_fedex_pricing', 'price', 'Shipping price in USD'),

  -- datasource_fedex_zone_distance_mapping
  ('datasource_fedex_zone_distance_mapping', 'id', 'Primary key, unique row identifier'),
  ('datasource_fedex_zone_distance_mapping', 'zone', 'FedEx shipping zone code'),
  ('datasource_fedex_zone_distance_mapping', 'min_distance', 'Minimum distance in miles for this zone'),
  ('datasource_fedex_zone_distance_mapping', 'max_distance', 'Maximum distance in miles for this zone'),

  -- shipping_rates
  ('shipping_rates', 'id', 'Primary key, unique row identifier'),
  ('shipping_rates', 'origin_zip', 'Origin ZIP code'),
  ('shipping_rates', 'destination_zip', 'Destination ZIP code'),
  ('shipping_rates', 'weight', 'Package weight in pounds'),
  ('shipping_rates', 'service_type', 'Shipping service type'),
  ('shipping_rates', 'carrier', 'Shipping carrier (e.g., FedEx, UPS, USPS)'),
  ('shipping_rates', 'price', 'Shipping price in USD'),
  ('shipping_rates', 'transit_days', 'Estimated transit days'),
  ('shipping_rates', 'created_at', 'Record creation timestamp'),
  ('shipping_rates', 'updated_at', 'Record last update timestamp'),

  -- shipping_zones
  ('shipping_zones', 'id', 'Primary key, unique row identifier'),
  ('shipping_zones', 'zone_code', 'Shipping zone code'),
  ('shipping_zones', 'zone_name', 'Shipping zone name'),
  ('shipping_zones', 'description', 'Description of the shipping zone'),
  ('shipping_zones', 'created_at', 'Record creation timestamp'),

  -- weather_data
  ('weather_data', 'id', 'Primary key, unique row identifier'),
  ('weather_data', 'location', 'Location name or code'),
  ('weather_data', 'temperature', 'Temperature in degrees Fahrenheit'),
  ('weather_data', 'humidity', 'Relative humidity percentage'),
  ('weather_data', 'wind_speed', 'Wind speed in miles per hour'),
  ('weather_data', 'precipitation', 'Precipitation in inches'),
  ('weather_data', 'weather_condition', 'Weather condition description'),
  ('weather_data', 'timestamp', 'Data record timestamp'),

  -- fuel_prices
  ('fuel_prices', 'id', 'Primary key, unique row identifier'),
  ('fuel_prices', 'fuel_type', 'Type of fuel (e.g., diesel, gasoline)'),
  ('fuel_prices', 'price_per_gallon', 'Fuel price per gallon in USD'),
  ('fuel_prices', 'region', 'Geographic region for fuel price'),
  ('fuel_prices', 'timestamp', 'Data record timestamp'),

  -- traffic_data
  ('traffic_data', 'id', 'Primary key, unique row identifier'),
  ('traffic_data', 'route', 'Route description or identifier'),
  ('traffic_data', 'congestion_level', 'Traffic congestion level'),
  ('traffic_data', 'average_speed', 'Average speed in miles per hour'),
  ('traffic_data', 'delay_minutes', 'Delay in minutes'),
  ('traffic_data', 'timestamp', 'Data record timestamp'),

  -- table_descriptions
  ('table_descriptions', 'table_name', 'Name of the table being described'),
  ('table_descriptions', 'column_name', 'Name of the column being described'),
  ('table_descriptions', 'description', 'Description of the column'); 