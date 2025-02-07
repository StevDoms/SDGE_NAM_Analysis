from geopy.distance import geodesic
from shapely.geometry import Point
import requests
from typing import Optional

def haversine_distance(point1: Point, point2: Point) -> float:
    """Computes Haversine distance between two shapely Point geometries."""
    lat1, lon1 = point1.y, point1.x
    lat2, lon2 = point2.y, point2.x
    return geodesic((lat1, lon1), (lat2, lon2)).km

def create_point(longitude: float, latitude: float) -> Point:
    return Point(longitude, latitude)

def get_elevation(longitude: float, latitude: float) -> Optional[float]:
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
    
    try:
        response = requests.get(url)
        
        # Check if the response status is successful
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return None
        
        # Ensure the response content is not empty before parsing as JSON
        if not response.text.strip():  # Check if response is empty
            print("Error: Empty response received.")
            return None
        
        # Try to parse the response as JSON
        response_json = response.json()
        
        # Return elevation if the 'results' key is found in the response
        return response_json['results'][0]['elevation'] if "results" in response_json else None
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except ValueError as e:
        print(f"JSON decode error: {e}")
        return None
