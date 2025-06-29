-- fedex_zone_distance_mapping_inserts.sql
INSERT INTO datasource_fedex_zone_distance_mapping (zone, min_distance, max_distance) VALUES
  ('2', 0, 150),
  ('3', 151, 300),
  ('4', 301, 600),
  ('5', 601, 1000),
  ('6', 1001, 1400),
  ('7', 1401, 1800),
  ('8', 1801, 2700); 