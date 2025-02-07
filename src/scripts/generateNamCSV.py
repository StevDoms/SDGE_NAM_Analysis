import requests
from requests.models import Response
from bs4 import BeautifulSoup
import os
import io
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional

def url_request(url: str, stream: bool = False) -> Response:
    """Sends a GET request to the given URL and returns the response."""
    response = requests.get(url, stream=stream)
    return response

def create_bs(response: Response) -> Optional[BeautifulSoup]:
    """Creates a BeautifulSoup object from the response HTML content."""
    if response.status_code == 200:
        html_content = response.text
    else:
        print(f"Failed to retrieve the page: {response.status_code}")
        return None  # Ensure function returns None in case of failure

    soup = BeautifulSoup(html_content, "html.parser")
    return soup

def find_all_element(soup: BeautifulSoup, element: str, element_class: str) -> List[BeautifulSoup]:
    """Finds all elements of a given type with a specified class in the BeautifulSoup object."""
    elements = soup.find_all(element, class_=element_class)
    return elements

def find_all_href(soup: BeautifulSoup, element: str, element_class: str) -> List[str]:
    """Finds all href attributes within elements of a given type and class."""
    elements = []
    td_elements = find_all_element(soup, element, element_class)[1:]

    for td in td_elements:
        a_tag = td.find("a")
        if a_tag and a_tag.get("href"):
            href = a_tag["href"]
            elements.append(href)

    return elements

def unique_dates(wind_speed_path: str) -> List[datetime]:
    windspeed = pd.read_csv(wind_speed_path)
    
    if 'date' not in windspeed.columns:
        raise ValueError("The CSV file must contain a 'date' column.")

    dates = windspeed['date'].dropna().astype(str)  # Ensure dates are strings
    unique_dates = dates.unique().tolist()
    unique_dates = [datetime.strptime(date, '%m/%d/%Y') for date in unique_dates]

    return unique_dates

def get_df(file_url: str) -> pd.DataFrame:
    # Getting file from SDGE website
    file_response = url_request(file_url)
    file_response.raise_for_status()

    file_obj = io.BytesIO(file_response.content)
    dataset = xr.open_dataset(file_obj)
    grid_data = dataset.to_dataframe()

    time = dataset['time'].values[0]  
    date = [time] * grid_data.shape[0]
    grid_data['date'] = date

    grid_data['wind_speed'] = np.sqrt(grid_data['eastward_925mb_wind']**2 + grid_data['northward_925mb_wind']**2)
    filtered_data = grid_data[['latitude', 'longitude', 'date', 'wind_speed']]
    filtered_data = filtered_data.dropna(axis=1, how='all')

    return filtered_data

def generate_nam_csv(wind_speed_path: str):
    # Starting page
    url = "https://sdge.sdsc.edu/data/sdge/historical-ens_gfs_004/portal/"
    
    # Filter unique dates from wind speed data
    wind_speed_unique_dates = unique_dates(wind_speed_path)
    
    # Root Directory Scraping
    base_response = url_request(url)
    base_soup = create_bs(base_response)
    
    # Get subdirectory elements
    subdirectory = find_all_href(base_soup, "td", "indexcolname")
    subdirectory = subdirectory[20:] # Filter to start from 2012
    
    # Store all output data
    output_wind_speed_df = pd.DataFrame(columns=['latitude', 'longitude', 'date', 'average_wind_speed'])
    
    # Download Files by going through each subdirectory
    for sub in subdirectory:
        subdir_url = url + sub
    
        subdir_response = url_request(subdir_url)
        subdir_soup = create_bs(subdir_response)
    
        file_name = find_all_href(subdir_soup, "td", "indexcolname")
        file_name = [file[2:] for file in file_name]
    
        print("Subdirectory: {}".format(sub))
    
        # Keep track of date being tracked for daily wind speed average
        current_file_date = None
        wind_speed_df_by_date = None
        
        # Reading each individual files
        for index, file in enumerate(file_name):
            file_date = file.split('_')[2]
            file_date = datetime.strptime(file_date, '%Y-%m-%d')
            file_time = file.split('_')[3]
            file_url = subdir_url + file
    
            next_file_date = file_name[index + 1].split('_')[2] if index != len(file_name) - 1 else file_name[index].split('_')[2]
            next_file_date = datetime.strptime(next_file_date, '%Y-%m-%d')
    
            if file_date not in wind_speed_unique_dates:
                print("File skipped: {}".format(file))
                continue
    
            if current_file_date == None or current_file_date != file_date:
                current_file_date = file_date
                wind_speed_df_by_date = pd.DataFrame(columns=['latitude', 'longitude', 'date', 'wind_speed'])
                print("Start Processing File Date: {}".format(current_file_date.strftime('%Y-%m-%d')))
    
            print("File processed: {}".format(file))
            file_df = get_df(file_url)
    
            if wind_speed_df_by_date.shape[0] == 0:
                wind_speed_df_by_date = file_df
            else:
                file_df = file_df.drop(columns=['date']).rename(columns={'wind_speed': f"wind_speed_{file_time}"})
    
                wind_speed_df_by_date = wind_speed_df_by_date.merge(
                    file_df, 
                    left_on=['latitude', 'longitude'], 
                    right_on=['latitude', 'longitude']
                )
    
            if index == len(file_name) - 1 or next_file_date != current_file_date:
                wind_speed_df_by_date_columns = list(wind_speed_df_by_date.columns)
                columns_to_remove = ['latitude', 'longitude', 'date']
    
                # Get average wind speed
                wind_speed_df_by_date_columns = [col for col in wind_speed_df_by_date_columns if col not in columns_to_remove]
                wind_speed_df_by_date['average_wind_speed'] = wind_speed_df_by_date[wind_speed_df_by_date_columns].mean(axis=1)
    
                # Concat average wind speed daily to output df
                wind_speed_df_by_date_average = wind_speed_df_by_date[['latitude', 'longitude', 'date', 'average_wind_speed']]
                wind_speed_df_by_date_average = wind_speed_df_by_date_average.dropna(subset=['average_wind_speed']) # Drop empty wind speed
                output_wind_speed_df = pd.concat([output_wind_speed_df, wind_speed_df_by_date_average], ignore_index=True)
    
                print("File Date Appended to Output: {}".format(current_file_date.strftime('%Y-%m-%d')))
                print(f"Current Output Shape: {output_wind_speed_df.shape}")
    
    # Create file name and directory path
    save_dir = "data/modified"
    output_file_name = '{}.csv'.format("nam")
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the DataFrame
    output_wind_speed_df.to_csv(os.path.join(save_dir, output_file_name), index=False)
    print("File Saved: {}".format(os.path.join(save_dir, output_file_name)))
    
    print(f"Download Complete")

    return