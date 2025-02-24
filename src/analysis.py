import pandas as pd
import numpy as np
from typing import List
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import alphashape

def custom_groupby(df: pd.DataFrame, groupby_cols: List[str], agg_dict: dict) -> pd.DataFrame:
    """
    Generalized function to group by specified columns and aggregate data.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to group.
    groupby_cols (list): List of column names to group by.
    agg_dict (dict): Dictionary specifying the aggregation functions for each column.
                     Example: {'column_name': 'mean', 'other_column': 'sum'}
    
    Returns:
    pd.DataFrame: The grouped and aggregated DataFrame.
    """
    return df.groupby(groupby_cols).agg(agg_dict).reset_index()

def find_outliers_iqr(df: pd.DataFrame, column: str) -> List:
    """
    Identifies outliers in a numerical column using the IQR method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to check for outliers.

    Returns:
    pd.DataFrame: A DataFrame containing only the outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return [lower_bound, upper_bound]

def create_polygon_gdf(coords: list, method: str = "convex", alpha: float = 0.3, name: str = None) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame containing a polygon from a list of coordinates.
    
    Parameters:
    - coords: List of (longitude, latitude) tuples.
    - method: "convex" for Convex Hull, "concave" for Alpha Shape.
    - alpha: Alpha value for Concave Hull (smaller = tighter fit).
    - name: Name for the polygon (optional).
    
    Returns:
    - GeoDataFrame with the polygon and optional name column.
    """
    if len(coords) < 3:
        raise ValueError("At least 3 points are required to form a polygon.")

    if method == "convex":
        hull = ConvexHull(coords)
        polygon = Polygon([coords[i] for i in hull.vertices])
    elif method == "concave":
        polygon = alphashape.alphashape(coords, alpha)
    else:
        raise ValueError("Method must be 'convex' or 'concave'")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")
    
    # Optionally add a name column
    if name is not None:
        gdf["boundary_name"] = name

    return gdf

def create_polygons_from_geometries(geometries: tuple) -> gpd.GeoDataFrame:
    """
    Creates a concatenated GeoDataFrame containing polygons from a tuple of coordinates.
    
    Parameters:
    - geometries: Tuple of tuples containing coordinates and associated names.
    
    Returns:
    - GeoDataFrame containing the concatenated polygons.
    """
    # Create an empty list to store GeoDataFrames
    outlier_boundary_list = []

    # Loop through each geometry and create a GeoDataFrame
    for coords, name in geometries:
        gdf = create_polygon_gdf(coords, method="concave", alpha=0.3, name=name)
        outlier_boundary_list.append(gdf)

    # Concatenate all GeoDataFrames together
    outlier_boundary_gdf = gpd.GeoDataFrame(pd.concat(outlier_boundary_list, ignore_index=True), crs="EPSG:4326")
    
    return outlier_boundary_gdf


