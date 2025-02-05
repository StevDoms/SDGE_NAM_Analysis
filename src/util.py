from geopy.distance import geodesic
from shapely.geometry import Point

def haversine_distance(point1: Point, point2: Point) -> float:
    """Computes Haversine distance between two shapely Point geometries."""
    lat1, lon1 = point1.y, point1.x
    lat2, lon2 = point2.y, point2.x
    return geodesic((lat1, lon1), (lat2, lon2)).km

def create_point(longitude: float, latitude: float) -> Point:
    return Point(longitude, latitude)

def get_elevation(longitude: float, latitude: float) -> float:
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
    response = requests.get(url).json()
    return response['results'][0]['elevation'] if "results" in response else None

