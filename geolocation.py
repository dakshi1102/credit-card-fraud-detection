from geopy.geocoders import Nominatim

def get_location_from_coordinates(lat, lon):
    geolocator = Nominatim(user_agent="fraud_detection_app")
    location = geolocator.reverse(f"{lat}, {lon}", language='en')
    return location.address if location else "Unknown location"
