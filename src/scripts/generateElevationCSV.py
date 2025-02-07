import sys
import os
from typing import List
import geopandas as gpd
import pandas as pd

from src.data import orange_county_geometry, san_diego_county_geometry
from src.util import get_elevation

def generate_elevation_csv(gis_weather_station_gpd: gpd.GeoDataFrame, src_vri_snapshot_gpd: gpd.GeoDataFrame, nam_gpd: gpd.GeoDataFrame):

    # Define the directory path
    file_save_path = "./data/modified"
    
    # Ensure the directory exists
    os.makedirs(file_save_path, exist_ok=True)

    # Get elevation for each weather station
    gis_weather_station_gpd['station_elevation_m'] = gis_weather_station_gpd.apply(lambda row: get_elevation(row['geometry'].x, row['geometry'].y), axis=1)

    # Save weather station gpd as CSV file
    station_file_path = os.path.join(file_save_path, "gis_weather_station_with_elevation.csv")
    gis_weather_station_gpd.to_csv(station_file_path, index=False)

    print(f"Station elevation saved to {station_file_path}")

    # Filter the geometry column
    san_diego_county_gpd = san_diego_county_geometry()
    orange_county_gpd = orange_county_geometry()

    # Filter NAM data within all VRI Polygon, San Diego county, and Orange County
    nam_vri_gpd = gpd.sjoin(nam_gpd, src_vri_snapshot_gpd, how='inner', predicate='within').drop(columns=['index_right'])
    nam_san_diego_county_gpd = gpd.sjoin(nam_gpd, san_diego_county_gpd, how='inner', predicate='within').drop(columns=['index_right'])
    nam_orange_county_gpd = gpd.sjoin(nam_gpd, orange_county_gpd, how='inner', predicate='within').drop(columns=['index_right'])

    # Combined all unique filtered NAM data
    nam_vri_gpd = nam_vri_gpd[['latitude', 'longitude', 'date', 'average_wind_speed', 'geometry']]
    nam_filtered_gpd = gpd.GeoDataFrame(pd.concat([nam_vri_gpd, nam_san_diego_county_gpd, nam_orange_county_gpd], ignore_index=True)).drop_duplicates()

    # Get the unique geometry from the filtered data
    nam_filtered_geometry_unique = nam_filtered_gpd[['geometry']].drop_duplicates().reset_index(drop=True)

    # Get elevation for each unique filtered NAM points
    nam_filtered_geometry_unique['nam_elevation_m'] = nam_filtered_geometry_unique.apply(lambda row: get_elevation(row['geometry'].x, row['geometry'].y), axis=1)

    # Add elevation to NAM data
    nam_filtered_gpd = nam_filtered_gpd.merge(nam_filtered_geometry_unique, on='geometry', how='inner')

    # # Save file as CSV
    nam_file_path = os.path.join(file_save_path, "nam_with_elevation.csv")
    nam_filtered_gpd.to_csv(nam_file_path, index=False)

    print(f"NAM elevation saved to {nam_file_path}")
    
    return

    

    


