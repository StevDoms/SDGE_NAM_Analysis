import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
import branca
import folium
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster
from shapely import wkt
from shapely.geometry import Point, Polygon, MultiPolygon
import branca
from IPython.display import IFrame
import alphashape

from typing import Union

def plot_data(data, x, y=None, plot_type="hist", bins=30, title=None, xlabel=None, ylabel=None, 
              color="blue", second_data=None, second_x=None, second_color="red", alpha=0.5, 
              hist_order="first", kde=False):
    """
    Generalized function to plot different types of visualizations, supporting dual histograms and KDE lines.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        x (str): The primary column for the x-axis (or the main column for histograms).
        y (str, optional): The column name for the y-axis (used in scatter, line, and bar plots).
        plot_type (str): Type of plot - "hist", "scatter", "line", "bar".
        bins (int): Number of bins for histograms. Default is 30.
        title (str): The title of the plot. Default is None.
        xlabel (str): The label for the x-axis. Default is the column name.
        ylabel (str): The label for the y-axis. Default is "Frequency" for histograms.
        color (str): The color of the plot elements. Default is "blue".
        second_data (pd.DataFrame, optional): A second DataFrame for overlaying histograms.
        second_x (str, optional): A column from the second DataFrame for the second histogram.
        second_color (str): The color of the second histogram. Default is "red".
        alpha (float): Transparency level for overlapping histograms. Default is 0.5.
        hist_order (str): Which histogram is in front, "first" or "second". Default is "first".
        kde (bool): Whether to overlay a Kernel Density Estimate (KDE) line. Default is False.
    """
    if x not in data.columns or (y and y not in data.columns):
        raise ValueError("Specified column(s) not found in the DataFrame.")

    plt.figure(figsize=(10, 6))

    # Handling second histogram data
    second_hist_data = None
    if second_data is not None and second_x is not None:
        if second_x not in second_data.columns:
            raise ValueError(f"Column '{second_x}' not found in second DataFrame.")
        second_hist_data = second_data[second_x]
    elif second_x is not None:
        if second_x not in data.columns:
            raise ValueError(f"Column '{second_x}' not found in the primary DataFrame.")
        second_hist_data = data[second_x]

    # Determine which histogram to plot first
    if plot_type == "hist":
        if hist_order == "second" and second_hist_data is not None:
            plt.hist(second_hist_data, bins=bins, color=second_color, edgecolor="black", alpha=alpha, label=second_x)
            plt.hist(data[x], bins=bins, color=color, edgecolor="black", alpha=alpha, label=x)
        else:
            plt.hist(data[x], bins=bins, color=color, edgecolor="black", alpha=alpha, label=x)
            if second_hist_data is not None:
                plt.hist(second_hist_data, bins=bins, color=second_color, edgecolor="black", alpha=alpha, label=second_x)
        
        ylabel = ylabel if ylabel else "Frequency"

        # KDE Overlay
        if kde:
            sns.kdeplot(data[x], color=color, linewidth=2, label=f"{x} KDE")
            if second_hist_data is not None:
                sns.kdeplot(second_hist_data, color=second_color, linewidth=2, label=f"{second_x} KDE")

    elif plot_type == "scatter":
        if y is None:
            raise ValueError("Scatter plot requires both x and y columns.")
        plt.scatter(data[x], data[y], color=color, alpha=0.7)

        # Fit a line of best fit (linear regression)
        coeffs = np.polyfit(data[x], data[y], 1)
        trendline = np.poly1d(coeffs)
    
        # Plot the trendline
        x_range = np.linspace(data[x].min(), data[x].max(), 100)
        plt.plot(x_range, trendline(x_range), color='red', linestyle='--', label="Best fit line")
    
        plt.legend()
    
    elif plot_type == "line":
        if y is None:
            raise ValueError("Line plot requires both x and y columns.")
        
        # Convert x-axis to datetime if applicable
        if pd.api.types.is_datetime64_any_dtype(data[x]) or data[x].dtype == 'object':
            data[x] = pd.to_datetime(data[x])

        sns.lineplot(x=x, y=y, data=data, color=color)

        # Format the x-axis for datetime plots
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Major locator: Year
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format: Year

        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())  # Minor locator: Month
    
    elif plot_type == "bar":
        if y is None:
            raise ValueError("Bar plot requires both x and y columns.")
        plt.bar(data[x], data[y], color=color, alpha=0.7)

    else:
        raise ValueError("Invalid plot_type. Choose from 'hist', 'scatter', 'line', or 'bar'.")

    # Labels and Title
    plt.title(title if title else f"{plot_type.capitalize()} Plot of {x}", fontsize=14)
    plt.xlabel(xlabel if xlabel else x, fontsize=12)
    plt.ylabel(ylabel if ylabel else "Frequency", fontsize=12)
    
    # Show legend only if we have multiple histograms or KDE
    if plot_type == "hist" and (second_x is not None or kde):
        plt.legend()
    
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_kde(data: pd.DataFrame, nam_col: str, station_col: str):
    """Plots the KDE distribution of NAM wind speed vs. weather station wind speed."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data[nam_col], fill=True, label="NAM Wind Speed", alpha=0.5)
    sns.kdeplot(data[station_col], fill=True, label="Weather Station Wind Speed", alpha=0.5)
    
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Density")
    plt.title("Distribution of NAM Wind Speed vs. Weather Station Wind Speed")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(data, method="pearson", title="Correlation Matrix", cmap="coolwarm", annot=True):
    """
    Plots the correlation matrix of a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing numerical data.
        method (str): The correlation method - "pearson", "kendall", or "spearman". Default is "pearson".
        title (str): The title of the heatmap. Default is "Correlation Matrix".
        cmap (str): Colormap for the heatmap. Default is "coolwarm".
        annot (bool): Whether to display correlation values in the heatmap. Default is True.
    """
    plt.figure(figsize=(8, 6))
    correlation_matrix = data.corr(method=method)
    sns.heatmap(correlation_matrix, annot=annot, fmt=".2f", cmap=cmap, linewidths=0.5, square=True)
    
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_map(weather_station: gpd.GeoDataFrame, vri_snapshot: gpd.GeoDataFrame, nam: Union[gpd.GeoDataFrame, pd.DataFrame], nam_color_column: str, output_file_name: str, show_alpha_shapes: bool = False, alpha: float = 1.0, cluster_centeroid: np.array = np.array([])):
    # Initialize the map centering at San Diego City
    m = folium.Map(location=[32.7157, -117.1611], zoom_start=9, tiles="OpenStreetMap")

    # Cluster Boundaries Group (if enabled)
    if show_alpha_shapes:
        cluster_boundaries = get_alpha_shapes(nam, alpha)
        cluster_group = folium.FeatureGroup(name="Cluster Boundaries")
        
        for _, row in cluster_boundaries.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda x: {
                    "fillColor": "#3388ff",
                    "color": "blue",
                    "weight": 2,
                    "fillOpacity": 0.3
                },
                tooltip=f"Cluster {row['cluster']}"
            ).add_to(cluster_group)

        cluster_group.add_to(m)  # Add cluster boundaries to the map

    if len(cluster_centeroid) > 0:
        centeroid_group = folium.FeatureGroup(name="Centeroid")
        
        for index, row in enumerate(cluster_centeroid):
            latitude, longitude = row[1], row[0]
            
            folium.CircleMarker(
                location=(latitude, longitude),
                radius=3,
                color='purple',
                fill=True,
                fill_color='purple', 
                fill_opacity=0.9,  
                opacity=0.9,    
                tooltip=(
                    f"Cluster: {index}<br>"
                    f"Coordinate: ({row[0]}, {row[1]})<br>"
                    f"Mean MAE: {row[2]:.3f}<br>"
            )
            ).add_to(centeroid_group)
    
    # NAM group
    nam_group = folium.FeatureGroup(name='nam')
    
    # Normalize the MAE values to ensure colors are mapped to a range
    min_value, max_value = nam[nam_color_column].min(), nam[nam_color_column].max()
    
    # Define colormap for yellow to red
    colormap = branca.colormap.LinearColormap(['#FFFF00', '#FF0000'], vmin=min_value, vmax=max_value)
    
    # Plotting the NAM points
    for _, row in nam.iterrows():
        latitude, longitude = row["geometry"].y, row["geometry"].x
        
        # Color based on the MAE value using the colormap
        color = colormap(row[nam_color_column])
        
        folium.CircleMarker(
            location=(latitude, longitude),
            radius=3,
            color=color,
            fill=True,
            fill_color=color, 
            fill_opacity=0.9,  
            opacity=0.9,    
            tooltip=(
                f"Coordinate: ({row['geometry'].x}, {row['geometry'].y})<br>"
                f"{nam_color_column}: {row[nam_color_column]:.3f}<br>"
        )
        ).add_to(nam_group)
    
    # Weather Station group
    weather_station_group = folium.FeatureGroup(name='weather_station_group')
    
    for idx, row in weather_station.iterrows():
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=4,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=1,
            opacity=1,
            tooltip=(f"Station: {row['weatherstationname']}<br>")
        ).add_to(weather_station_group)

    
    # VRI Snapshot
    vri_group = folium.FeatureGroup(name='vri_group')
    
    # Load simplified GeoJSON with tooltip
    vri_tooltip = folium.GeoJsonTooltip(
        fields=["name", "vri_risk", "shape_area"],
        aliases=["Name:", "VRI Risk:", "Shape Area:"],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    
    # Load VRI GeoJSON
    vri_points = folium.GeoJson(
        vri_snapshot,
        style_function=lambda x: {
            "fillColor": "#0059b3",
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.5
        },
        tooltip=vri_tooltip,
    )
    
    vri_points.add_to(vri_group)
    
    # Add feature groups to the map
    vri_group.add_to(m)
    nam_group.add_to(m)
    weather_station_group.add_to(m)
    if len(cluster_centeroid) > 0:
        centeroid_group.add_to(m)  # Add cluster boundaries to the map
    
    # Add layer control to toggle feature groups
    folium.LayerControl().add_to(m)
    
    # Check directory exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Save Map
    map_path = os.path.join(plots_dir, output_file_name)
    m.save(map_path)

def get_alpha_shapes(df: pd.DataFrame, alpha: float = 1.0):
    """
    Compute alpha shapes for each cluster in the dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame with 'x', 'y', and 'cluster' columns.
        alpha (float): Alpha parameter for the alpha shape. Lower values result in smoother shapes.

    Returns:
        gpd.GeoDataFrame: Cluster boundaries as polygon geometries.
    """
    cluster_polygons = []

    for cluster_id, group in df.groupby("cluster"):
        if len(group) < 3:  # Alpha Shape requires at least 3 points
            continue
        
        points = np.array(group[["x", "y"]])
        alpha_shape = alphashape.alphashape(points, alpha)
        
        # Convert MultiPolygon to Polygon if needed
        if isinstance(alpha_shape, MultiPolygon):
            for poly in alpha_shape.geoms:
                cluster_polygons.append({"cluster": cluster_id, "geometry": poly})
        else:
            cluster_polygons.append({"cluster": cluster_id, "geometry": alpha_shape})
    
    return gpd.GeoDataFrame(cluster_polygons, geometry="geometry")

