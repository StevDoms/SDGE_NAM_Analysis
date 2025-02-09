import sys
import os
import json

from src.model import light_gbm, predict_light_gbm_model
from src.data import generate_df, generate_gdf, preprocess_df, preprocess_gdf, filter_nam_outside_vri, get_nam_outside_vri_nearest_station
from src.scripts.generateNamCSV import generate_nam_csv
from src.scripts.generateElevationCSV import generate_elevation_csv

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'features', 'model'.

    `main` runs the targets in order of data=>analysis=>model.
    '''

    with open('config/data_params.json') as fh:
        data_params = json.load(fh)    

    raw_data_path = [os.path.join('./data/raw', file_path) for file_path in data_params["raw_data"]]
    modified_data_path = [os.path.join('./data/modified', file_path) for file_path in data_params["modified_data"]]
    output_model_path = [os.path.join('./data/modified', file_path) for file_path in data_params["model_prediction"]]

    if "create_nam_file" in targets:
        wind_speed_path = raw_data_path[3]
        generate_nam_csv(wind_speed_path)

    if "create_elevation_file" in targets:
        print(raw_data_path)
        gis_weather_station, src_vri_snapshot, nam, windspeed_snapshot = generate_df(raw_data_path) 
        gis_weather_station, windspeed_snapshot = preprocess_df(gis_weather_station, windspeed_snapshot)
        gis_weather_station_gpd, src_vri_snapshot_gpd, nam_gpd = generate_gdf(gis_weather_station, src_vri_snapshot, nam)
        
        generate_elevation_csv(gis_weather_station_gpd, src_vri_snapshot_gpd, nam_gpd)
        
    if "process_model_input" in targets:
        model_input_file_path = modified_data_path + [raw_data_path[1], raw_data_path[3]]
        gis_weather_station_with_elevation, nam_with_elevation, src_vri_snapshot, windspeed_snapshot = generate_df(model_input_file_path)
        gis_weather_station_with_elevation, windspeed_snapshot = preprocess_df(gis_weather_station_with_elevation, windspeed_snapshot)
        gis_weather_station_with_elevation_gpd, src_vri_snapshot_gpd, nam_with_elevation_gpd = generate_gdf(gis_weather_station_with_elevation, src_vri_snapshot, nam_with_elevation)

        model_input_gdf = preprocess_gdf(gis_weather_station_with_elevation_gpd, src_vri_snapshot_gpd, nam_with_elevation_gpd, windspeed_snapshot)

    if "light_gbm_model" in targets:
        light_gbm_model = light_gbm(model_input_gdf) # Fitting the lightgbm model

    if "predict_model" in targets:
        # Predicting error of NAM data within the VRI polygons (training data)
        nam_within_vri_prediction = predict_light_gbm_model(light_gbm_model, model_input_gdf)
        nam_within_vri_prediction.to_csv(output_model_path[0], index=False)

        # Predicting error of NAM data outside the VRI polygons (unseen data)
        nam_outside_vri = filter_nam_outside_vri(nam_with_elevation_gpd, model_input_gdf)
        nam_outside_vri_nearest_station = get_nam_outside_vri_nearest_station(nam_outside_vri, gis_weather_station_with_elevation_gpd)
        nam_outside_vri_prediction = predict_light_gbm_model(light_gbm_model, nam_outside_vri_nearest_station)
        nam_outside_vri_prediction.to_csv(output_model_path[1], index=False)

    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)