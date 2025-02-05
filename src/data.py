import pandas as pd
import geopandas as gpd
from shapely import wkt
from typing import List, Tuple
from src.util import create_point

def read_df(input_list: List[str]) -> List[pd.DataFrame]:
    gis_weather_station = pd.read_csv(input_list[0])
    src_vri_snapshot = pd.read_csv(input_list[1])
    nam = pd.read_csv(input_list[2])
    san_diego_county_gpd = gpd.read_file(input_list[3])
    windspeed_snapshot = pd.read_csv(input_list[4])

    return [gis_weather_station, src_vri_snapshot, nam, san_diego_county_gpd, windspeed_snapshot]
    

def generate_gdf(df_list: List[pd.DataFrame]) -> List[gpd.GeoDataFrame]:
    gis_weather_station, src_vri_snapshot, nam, san_diego_county_gpd = df_list

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
    print(f"San Diego County:       {san_diego_county_gpd.crs}")

    return [gis_weather_station_gpd, src_vri_snapshot_gpd, nam_gpd, san_diego_county_gpd]

def orange_county_geometry():
    orange_county_geometry = gpd.read_file("data/raw/Congressional.shp").iloc[[5]] # Southern Orange County
    orange_county_gpd = gpd.GeoDataFrame(orange_county_geometry, geometry='geometry', crs=f"EPSG:4326") 
    return orange_county_gpd


    





