import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from typing import List, Tuple
from src.util import create_point, haversine_distance

def generate_df(input_list: List[str]) -> List[pd.DataFrame]:
    return [pd.read_csv(file) for file in input_list]

def convert_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=f"ESPG:4326")
    return gdf
    
def generate_gdf(gis_weather_station: pd.DataFrame, src_vri_snapshot: pd.DataFrame, nam: pd.DataFrame) -> List[gpd.GeoDataFrame]:

    # EPSG:4431
    gis_weather_station['geometry'] = gis_weather_station['shape'].apply(wkt.loads)
    gis_weather_station_gpd = gpd.GeoDataFrame(gis_weather_station, geometry='geometry', crs=f"EPSG:{gis_weather_station['shape_srid'][0]}")

    # ESPG:4326
    src_vri_snapshot['geometry'] = src_vri_snapshot['shape'].apply(wkt.loads)
    src_vri_snapshot_gpd = gpd.GeoDataFrame(src_vri_snapshot, geometry='geometry', crs=f"EPSG:{src_vri_snapshot['shape_srid'][0]}")
    
    # ESPG:4326
    nam_crs = src_vri_snapshot['shape_srid'][0]
    nam['geometry'] = nam.apply(lambda row: create_point(row['longitude'], row['latitude']), axis=1)
    nam_gpd = gpd.GeoDataFrame(nam, geometry='geometry', crs=nam_crs)

    # Convert EPSG:4431 to ESPG:4326
    gis_weather_station_gpd = gis_weather_station_gpd.to_crs(src_vri_snapshot_gpd.crs)

    print(f"Weather Station CRS:    {gis_weather_station_gpd.crs}")
    print(f"VRI Polygon CRS:        {src_vri_snapshot_gpd.crs}")
    print(f"NAM CRS:                {nam_gpd.crs}")

    return [gis_weather_station_gpd, src_vri_snapshot_gpd, nam_gpd]

def orange_county_geometry():
    orange_county_geometry = gpd.read_file("./data/raw/Congressional.shp").iloc[[5]] # Southern Orange County
    orange_county_gpd = gpd.GeoDataFrame(orange_county_geometry, geometry='geometry', crs=f"EPSG:4326") 
    return orange_county_gpd

def san_diego_county_geometry():
    san_diego_county_geometry = gpd.read_file("./data/raw/San_Diego_County_Boundary.geojson")
    san_diego_county_gpd = san_diego_county_geometry[['geometry']] 
    return san_diego_county_gpd

def preprocess_df(gis_weather_station: pd.DataFrame, windspeed_snapshot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gis_weather_station = gis_weather_station.drop_duplicates(subset=['weatherstationcode'], keep='first')
    windspeed_snapshot = windspeed_snapshot[windspeed_snapshot['wind_speed'] < windspeed_snapshot['wind_speed'].max()]
    return gis_weather_station, windspeed_snapshot

def preprocess_gdf(gis_weather_station_gpd: gpd.GeoDataFrame, src_vri_snapshot_gpd: gpd.GeoDataFrame, nam_gpd: gpd.GeoDataFrame, windspeed_snapshot: pd.DataFrame) -> gpd.GeoDataFrame: 
    
    # Merge weather station with wind speed data
    weather_station_wind_speed_gpd = gis_weather_station_gpd.merge(windspeed_snapshot, left_on='weatherstationcode', right_on='station', 
                                                         how='inner').drop(columns=['station'])

    # Filter NAM data within VRI polygon
    nam_vri_gpd = gpd.sjoin(nam_gpd, src_vri_snapshot_gpd, how='right', predicate='within') # Join keeps the VRI polygon geometry
    nam_vri_gpd['nam_geometry'] = nam_vri_gpd.apply(lambda row: create_point(row['longitude'], row['latitude']), axis=1) # Creates NAM geometry
    nam_vri_gpd = nam_vri_gpd.reset_index(drop=True).drop(columns=['index_left'])

    # Combine weather station data with NAM data
    nam_vri_wind_speed_gpd = gpd.sjoin(weather_station_wind_speed_gpd, nam_vri_gpd, how='inner', predicate='within') # Join keeps the weather station geometry
    nam_vri_wind_speed_gpd = nam_vri_wind_speed_gpd.reset_index(drop=True).drop(columns=['index_right'])

    # Rename columns
    nam_vri_wind_speed_gpd = nam_vri_wind_speed_gpd.rename(columns={
        'date_left' : 'station_date',
        'date_right': 'nam_date',
        'wind_speed': 'station_wind_speed',
        'average_wind_speed' : 'nam_wind_speed',
        'shape_right': 'polygon_shape',
        # 'elevation_m': 'station_elevation_m'
    })

    # Change date to standardize date formatting
    nam_vri_wind_speed_gpd['nam_date'] = pd.to_datetime(nam_vri_wind_speed_gpd['nam_date']).dt.strftime('%m/%d/%Y')

    # Filter NAM data and wind speed data with the same dates
    filtered_nam_vri_wind_speed_gpd = nam_vri_wind_speed_gpd[nam_vri_wind_speed_gpd['station_date'] == nam_vri_wind_speed_gpd['nam_date']].copy()

    # Create new columns
    filtered_nam_vri_wind_speed_gpd['station_geometry'] = filtered_nam_vri_wind_speed_gpd['geometry']
    filtered_nam_vri_wind_speed_gpd['polygon_geometry'] = filtered_nam_vri_wind_speed_gpd['polygon_shape'].apply(wkt.loads)
    filtered_nam_vri_wind_speed_gpd['nam_distance_from_station_km'] = filtered_nam_vri_wind_speed_gpd.apply(
        lambda row: haversine_distance(row['station_geometry'], row['nam_geometry']), axis=1
    )

    # Get month and doy for temporal features
    filtered_nam_vri_wind_speed_gpd['month'] = pd.to_datetime(filtered_nam_vri_wind_speed_gpd['station_date']).dt.month
    filtered_nam_vri_wind_speed_gpd['day_of_year'] = pd.to_datetime(filtered_nam_vri_wind_speed_gpd['station_date']).dt.dayofyear
    filtered_nam_vri_wind_speed_gpd = filtered_nam_vri_wind_speed_gpd.reset_index(drop=True)

    grouped_columns = ["nam_geometry"]

    filtered_nam_vri_wind_speed_gpd["abs_wind_speed_error"] = (
        filtered_nam_vri_wind_speed_gpd["nam_wind_speed"] - filtered_nam_vri_wind_speed_gpd["station_wind_speed"]
    ).abs()

    return filtered_nam_vri_wind_speed_gpd

def filter_nam_outside_vri(nam_gpd: gpd.GeoDataFrame, model_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Get DataFrame subset
    nam_gpd_subset = nam_gpd[['latitude', 'longitude', 'date', 'average_wind_speed', 'geometry', 'nam_elevation_m']].rename(columns={'average_wind_speed': 'nam_wind_speed', 'date': 'nam_date'})
    model_gdf_subset = model_gdf[['latitude_right', 'longitude_right']]
    
    # Merge based on latitude and longitude, specifying left and right columns
    merged_df = pd.merge(nam_gpd_subset, model_gdf_subset, left_on=['latitude', 'longitude'], right_on=['latitude_right', 'longitude_right'], how='left', indicator=True)
    
    # Filter rows where there is no match in model_gdf (i.e., rows from nam_gpd only)
    nam_not_in_vri = merged_df[merged_df['_merge'] == 'left_only']
    
    # Reset the index and drop the merge indicator column
    nam_not_in_vri = nam_not_in_vri.reset_index(drop=True).drop(columns=['latitude_right', 'longitude_right', '_merge'])
    nam_not_in_vri['month'] = pd.to_datetime(nam_not_in_vri['nam_date']).dt.month
    nam_not_in_vri['day_of_year'] = pd.to_datetime(nam_not_in_vri['nam_date']).dt.dayofyear
    
    return nam_not_in_vri

def get_nam_outside_vri_nearest_station(nam_outside_vri: gpd.GeoDataFrame, gis_weather_station_gpd: gpd.GeoDataFrame) -> pd.DataFrame:

    # Get the unique geometry
    nam_outside_vri_unique_geometry = nam_outside_vri[['geometry']].drop_duplicates().reset_index(drop=True)

    nam_geometries = []
    nearest_station_geometry = []
    nearest_station_elevation = []
    nearest_station_distances = []
    
    for _, row in nam_outside_vri_unique_geometry.iterrows():
        nam_geometry = row["geometry"]

        # To capture output
        min_distance = float("inf")
        nearest_geometry = None
        nearest_elevation = None
        
        for _, other_row in gis_weather_station_gpd.iterrows():
            station_geometry = other_row["geometry"]
            station_elevation = other_row['station_elevation_m']   
            distance = haversine_distance(nam_geometry, station_geometry)
            
            if distance < min_distance:
                min_distance = distance
                nearest_geometry = station_geometry
                nearest_elevation = station_elevation

        nam_geometries.append(nam_geometry)
        nearest_station_geometry.append(nearest_geometry)
        nearest_station_elevation.append(nearest_elevation)
        nearest_station_distances.append(min_distance)
    
    output_df = pd.DataFrame({
        'geometry': nam_geometries,
        'station_geometry': nearest_station_geometry,
        'station_elevation_m': nearest_station_elevation,
        'nam_distance_from_station_km': nearest_station_distances
    })

    model_data = nam_outside_vri.merge(output_df, on='geometry', how='inner')

    return model_data





    





