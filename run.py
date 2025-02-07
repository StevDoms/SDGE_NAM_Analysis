import sys
import os
import json

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

    if "create_nam_file" in targets:
        wind_speed_path = raw_data_path[4]
        generate_nam_csv(wind_speed_path)

    if "create_elevation_file" in targets:
        raw_df = generate_df(raw_data_path)
        gis_weather_station_gpd, src_vri_snapshot_gpd, nam_gpd = generate_gdf(raw_df[0:2])
        generate_elevation_csv([gis_weather_station_gpd, src_vri_snapshot_gpd, nam_gpd, raw_df[3]])
        
    if "preprocess_data" in targets:
        
        

    if "light_gbm_model" in targets:

    
    if "predict_model" in targets:
        

        

    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)