def plot_map()
# Initialize the map centering at San Diego City
m = folium.Map(location=[32.7157, -117.1611], zoom_start=10, tiles="OpenStreetMap")

# NAM Coordinates
NAM_coordinates = folium.FeatureGroup(name='NAM_coordinates')

# Normalize the MAE values to ensure colors are mapped to a range
min_mae, max_mae = errors["MAE"].min(), errors["MAE"].max()

# Define colormap for yellow to red
colormap = branca.colormap.LinearColormap(['#FFFF00', '#FF0000'], vmin=min_mae, vmax=max_mae)

# Plot each point on the map with constant opacity and color based on MAE
for _, row in errors.iterrows():
    latitude, longitude = row["NAM_geometry"].y, row["NAM_geometry"].x
    
    # Color based on the MAE value using the colormap
    color = colormap(row["MAE"])
    
    folium.CircleMarker(
        location=(latitude, longitude),
        radius=3,
        color=color,
        fill=True,
        fill_color=color, 
        fill_opacity=0.9,  
        opacity=0.9,    
        tooltip=(f"MAE: {row['MAE']:.3f}<br>"
                 f"MSE: {row['MSE']:.3f}<br>"
                 f"NMAE: {row['NMAE']:.3f}<br>"
                 f"NMSE: {row['NMSE']:.3f}<br>"
                 f"DWAE: {row['DWAE']:.3f}<br>"
                 f"SDWE: {row['SDWE']:.3f}<br>"
                 f"Dist: {row['distance_from_station_km']:.3f}km<br>"
    )
    ).add_to(NAM_coordinates)

# Weather Station
weather_stations = folium.FeatureGroup(name='Weather Stations')

for idx, row in weather_station_summary_gpd.iterrows():
    folium.CircleMarker(
        location=(row["latitude"], row["longitude"]),
        radius=4,
        color="green",
        fill=True,
        fill_color="green",
        fill_opacity=1,
        opacity=1,
        tooltip=(f"Station: {row['weatherstationname']}<br>")
    ).add_to(weather_stations)

# VRI Snapshot
vri_snapshot = folium.FeatureGroup(name='VRI Snapshot')

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
vri_map = folium.GeoJson(
    src_vri_snapshot_gpd,
    style_function=lambda x: {
        "fillColor": "#0059b3",
        "color": "black",
        "weight": 0.3,
        "fillOpacity": 0.5
    },
    tooltip=vri_tooltip,
)
vri_map.add_to(vri_snapshot)

# Add feature groups to the map
vri_snapshot.add_to(m)
NAM_coordinates.add_to(m)
weather_stations.add_to(m)

# Add layer control to toggle feature groups
folium.LayerControl().add_to(m)

# Save the map
map_path = "san_diego_map_MAE.html"
m.save(map_path)

# Render the map in the notebook using IFrame
IFrame(map_path, width=700, height=500)
