import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geopy.distance import geodesic
from typing import List, Tuple
from src.data import generate_df, convert_to_gdf, generate_gdf, preprocess_df
from src.plot import plot_data
from src.model import light_gbm, predict_light_gbm_model
from src.data import generate_df, generate_gdf, preprocess_df, preprocess_gdf, filter_nam_outside_vri, get_nam_outside_vri_nearest_station
from src.scripts.generateNamCSV import generate_nam_csv
from src.scripts.generateElevationCSV import generate_elevation_csv
from src.analysis import custom_groupby, find_outliers_iqr




def load_data(raw_data_path: str, modified_data_path: str, output_model_path: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    
    # Reading raw data
    gis_weather_station, src_vri_snapshot, nam, windspeed_snapshot = generate_df(raw_data_path) 
    gis_weather_station, windspeed_snapshot = preprocess_df(gis_weather_station, windspeed_snapshot)
    
    # Reading filtered data with elevation from API
    gis_weather_station_with_elevation, nam_with_elevation = generate_df(modified_data_path)
    gis_weather_station_with_elevation_gpd, src_vri_snapshot_gpd, nam_with_elevation_gpd = generate_gdf(
        gis_weather_station_with_elevation, src_vri_snapshot, nam_with_elevation)

    nam_within_vri_prediction, nam_outside_vri_prediction = generate_df(output_model_path)
    
    nam_within_vri_prediction_gpd = convert_to_gdf(nam_within_vri_prediction)
    
    return gis_weather_station_with_elevation_gpd, windspeed_snapshot, src_vri_snapshot_gpd, nam_within_vri_prediction_gpd

def get_nam_points_to_update(nam_within_vri_prediction_gpd: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates wind speed error difference and aggregates metrics for NAM points.

    Args:
        nam_within_vri_prediction_gpd (gpd.GeoDataFrame): Input GeoDataFrame containing NAM point wind speed errors.

    Returns:
        gpd.GeoDataFrame: Aggregated GeoDataFrame with mean error metrics and distance-weighted error.
    """
    # Compute wind speed error difference
    nam_within_vri_prediction_gpd['wind_speed_error_diff'] = (
        nam_within_vri_prediction_gpd['abs_wind_speed_error'] - 
        nam_within_vri_prediction_gpd['abs_wind_speed_error_pred']
    ).abs()

    # Define aggregation dictionary
    agg_dict_within = {
        'abs_wind_speed_error': 'mean',
        'wind_speed_error_diff': 'mean',
        'nam_distance_from_station_km': 'mean'
    }

    # Group the data based on each NAM point
    nam_mae_within = nam_within_vri_prediction_gpd.groupby('geometry', as_index=False).agg(agg_dict_within)

    # Compute distance-weighted error
    nam_mae_within['distance_weight_error'] = (
        nam_mae_within['abs_wind_speed_error'] * nam_mae_within['nam_distance_from_station_km']
    )

    nam_points_to_update = nam_mae_within.sort_values(
    by='abs_wind_speed_error', ascending=False
    ).head(20)['geometry'].tolist()
    nam_points_to_update
    

    return nam_points_to_update

def filter_nam_within_vri_prediction_gpd(
    nam_within_vri_prediction_gpd: gpd.GeoDataFrame, 
    nam_points_to_update: List[Point]
) -> gpd.GeoDataFrame:

    filtered_nam_within_vri = nam_within_vri_prediction_gpd[
        nam_within_vri_prediction_gpd['geometry'].isin(nam_points_to_update)
    ]

    return filtered_nam_within_vri

# Function to find the nearest alternative VRI polygon (excluding current)
def find_nearest_vri(nam_point, vri_polygons, current_vri_name):
    nam_coords = (nam_point.y, nam_point.x)  # (lat, lon)
    nearest_vri = None
    min_distance = float("inf")

    for _, row in vri_polygons.iterrows():
        if row['name'] == current_vri_name:
            continue  # Skip the current VRI polygon

        vri_centroid = row['geometry'].centroid
        vri_coords = (vri_centroid.y, vri_centroid.x)

        distance = geodesic(nam_coords, vri_coords).km  # Haversine distance in km
        if distance < min_distance:
            min_distance = distance
            nearest_vri = row

    return nearest_vri

def update_nam_polygons(src_vri_snapshot: gpd.GeoDataFrame, nam_within_vri_prediction_gpd: gpd.GeoDataFrame, vri_gdf: gpd.GeoDataFrame, nam_points_to_update: List[str]) -> gpd.GeoDataFrame:

    nam_within_vri_prediction_gpd["New_VRI_Anemometer"] = "no change"
    nam_within_vri_prediction_gpd["New_Polygon_Shape"] = "no change"

    # Apply nearest VRI assignment for the specific NAM points
    for point_str in nam_points_to_update:
        current_vri_info = nam_within_vri_prediction_gpd.loc[
            nam_within_vri_prediction_gpd['geometry'] == point_str,
            ['polygon_geometry', 'name']
        ]
    
        if current_vri_info.empty:
            continue  # Skip if no current VRI info found
    
        current_vri_name = current_vri_info.iloc[0]['name']
    
        # Find the nearest alternative VRI polygon (excluding current)
        nearest_vri = find_nearest_vri(point_str, src_vri_snapshot, current_vri_name)
    
        if nearest_vri is not None:
            nam_within_vri_prediction_gpd.loc[
                nam_within_vri_prediction_gpd['geometry'] == point_str,
                ["New_VRI_Anemometer", "New_Polygon_Shape"]
            ] = [nearest_vri['anemometer'], nearest_vri['shape']]
        

    return nam_within_vri_prediction_gpd

def merge_weather_windspeed(gis_weather_station: gpd.GeoDataFrame, windspeed_snapshot: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merges weather station data with wind speed snapshot data.

    Args:
        gis_weather_station (GeoDataFrame): Weather station data.
        windspeed_snapshot (GeoDataFrame): Wind speed snapshot data.

    Returns:
        GeoDataFrame: Merged dataset.
    """
    return gis_weather_station.merge(
        windspeed_snapshot,
        left_on=["weatherstationcode"],
        right_on=["station"],
        how="inner"
    )

def spatial_join_vri(merged_wind_data: gpd.GeoDataFrame, src_vri_snapshot: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Performs a spatial join between merged wind data and VRI polygons.

    Args:
        merged_wind_data (GeoDataFrame): Merged wind data.
        src_vri_snapshot (GeoDataFrame): VRI polygon data.

    Returns:
        GeoDataFrame: Spatially joined dataset.
    """
    return gpd.sjoin(merged_wind_data, src_vri_snapshot, predicate="within")


def calculate_error_metrics(nam_points_to_update: List[Point], nam_within_vri_prediction_gpd: gpd.GeoDataFrame, merged_wind_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates wind speed error metrics.

    Args:
        filtered_merged_wind_data (GeoDataFrame): Dataset containing NAM wind speeds and station wind speeds.

    Returns:
        GeoDataFrame: Updated dataset with error metrics.
    """
    df = nam_within_vri_prediction_gpd[['geometry', 'abs_wind_speed_error']]
    df_filtered = df[df['geometry'].isin(nam_points_to_update)]
    df_filtered.groupby('geometry').mean()
    
    filtered_nam_within_vri_subset = nam_within_vri_prediction_gpd[
    nam_within_vri_prediction_gpd['geometry'].isin(nam_points_to_update)]

    
    merged_filtered_wind_data = filtered_nam_within_vri_subset.merge(
        merged_wind_data,
        left_on=["New_VRI_Anemometer", "nam_date"],  # Columns from filtered_nam_within_vri_subset
        right_on=["anemometer", "date"],            # Columns from merged_wind_data
        how="left"  # Keep all rows from filtered_nam_within_vri_subset
    )

        # Select relevant columns for calculation
    important_columns = [
        "geometry_x",
        "nam_wind_speed",
        "New_VRI_Anemometer",
        "wind_speed"
    ]
    
    filtered_merged_wind_data = merged_filtered_wind_data[important_columns]
    
    # Filter to only include rows where New_VRI_Anemometer has changed
    filtered_merged_wind_data = filtered_merged_wind_data[
        filtered_merged_wind_data["New_VRI_Anemometer"] != "no change"
    ]
    
    # Calculate new absolute wind speed error
    filtered_merged_wind_data["new_abs_wind_speed_error"] = (
        filtered_merged_wind_data["nam_wind_speed"] - filtered_merged_wind_data["wind_speed"]
    ).abs()


        # Group both DataFrames by 'nam_geometry' and compute mean errors
    new_error_df = filtered_merged_wind_data.groupby("geometry_x")[["new_abs_wind_speed_error"]].mean().reset_index()
    old_error_df = df_filtered.groupby("geometry")[["abs_wind_speed_error"]].mean().reset_index()
    
    # Merge both DataFrames to compare old vs new error
    comparison_df = old_error_df.merge(new_error_df, left_on='geometry', right_on='geometry_x').drop('geometry_x', axis=1)
    
    # Calculate the difference (improvement)
    comparison_df["error_difference"] = comparison_df["abs_wind_speed_error"] - comparison_df["new_abs_wind_speed_error"]

        # Extract new VRI polygon information
    filtered_merged_wind_data.rename(columns={'geometry_x': 'geometry'}, inplace=True)
    new_vri_info = filtered_merged_wind_data[["geometry", "New_VRI_Anemometer"]].drop_duplicates()
    
    # Merge to include the new VRI polygons
    comparison_df = comparison_df.merge(new_vri_info, on="geometry", how="left")
    
    # Filter for optimal geometries where the error difference is positive (improvement)
    optimum_geometry = comparison_df[comparison_df["error_difference"] > 0]
    
    # Rename column for clarity
    optimum_geometry.rename(columns={"New_VRI_Anemometer": "New_VRI_Polygon"}, inplace=True)

    return optimum_geometry



