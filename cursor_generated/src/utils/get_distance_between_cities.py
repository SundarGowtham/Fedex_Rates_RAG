import os
import requests
from dotenv import load_dotenv
import googlemaps


# Load environment variables from .env file
load_dotenv()

# OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
# if not OPENROUTER_API_KEY:
#     raise EnvironmentError('OPENROUTER_API_KEY environment variable is not set.')

GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
if not GOOGLE_MAPS_API_KEY:
    raise EnvironmentError('GOOGLE_MAPS_API_KEY environment variable is not set.')


gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


def get_distance_between_cities(origin_city, destination_city):
    my_dist = gmaps.distance_matrix(origin_city, destination_city)['rows'][0]['elements'][0]

    # SAMPLE OUTPUT: {'distance': {'text': '4,733 km', 'value': 4733183}, 'duration': {'text': '1 day 19 hours', 'value': 155288}, 'status': 'OK'}

    # returns distance in km
    return my_dist['distance']['value'] / 1000




# def geocode_city(city_name):
#     """Geocode a city name to get [lon, lat]"""
#     geocode_url = 'https://api.openrouteservice.org/geocode/search'
#     params = {
#         'api_key': OPENROUTER_API_KEY,
#         'text': city_name
#     }
#     response = requests.get(geocode_url, params=params)
#     response.raise_for_status()
#     features = response.json().get('features')
#     if not features:
#         raise ValueError(f"Could not geocode city: {city_name}")
#     return features[0]['geometry']['coordinates']  # [lon, lat]

# def get_route_distance(origin_coords, dest_coords):
#     """Get distance and duration from ORS Directions API"""
#     url = 'https://api.openrouteservice.org/v2/directions/driving-car'
#     headers = {
#         'Authorization': OPENROUTER_API_KEY,
#         'Content-Type': 'application/json'
#     }
#     payload = {
#         'coordinates': [origin_coords, dest_coords]
#     }
#     response = requests.post(url, json=payload, headers=headers)
#     response.raise_for_status()
#     summary = response.json()['features'][0]['properties']['summary']
#     return summary['distance'] / 1000, summary['duration'] / 60  # km, minutes

# def get_distance_between_cities(origin_city, destination_city):
#     """Get the driving distance in km between two cities by name."""
#     origin_coords = geocode_city(origin_city)
#     dest_coords = geocode_city(destination_city)
#     distance_km, _ = get_route_distance(origin_coords, dest_coords)
#     return distance_km

# if __name__ == "__main__":
#     # CLI usage
#     origin_city = input("Enter origin city: ")
#     destination_city = input("Enter destination city: ")

#     try:
#         distance_km = get_distance_between_cities(origin_city, destination_city)
#         print(f"\n🚗 From {origin_city} to {destination_city}:")
#         print(f"Distance: {distance_km:.2f} km")
#     except Exception as e:
#         print("❌ Error:", e)