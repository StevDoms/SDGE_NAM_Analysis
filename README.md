# Analyzing Wind Speed Error Using the NAM Data

---

## Project Overview  

This project is divided into two main parts: **Model Training** and **Analysis**.  

## 1. Model Training  
To train the model, we follow these key steps:  
- **Obtain NAM Data** – Collect North American Mesoscale (NAM) model data.  
- **Retrieve Elevation Data** – Gather elevation data for both weather stations and NAM grid points.  
- **Preprocess Data** – Prepare the dataset for model input, including cleaning and feature engineering.  
- **Train the LightGBM Model** – Use LightGBM to train a predictive model.  
- **Make Predictions** – Utilize the trained model to generate forecasts.  

## 2. Analysis  
For analysis, we focus on:  
- **Exploratory Data Analysis (EDA)** – Visualizing and understanding the dataset.  
- **Error Assessment** – Evaluating the accuracy of NAM predictions by comparing them to actual measurements.  
- **Notebook Implementation** – Conducting the analysis in a Jupyter Notebook for better visualization and documentation.

---

## How to Run the Code

### Prerequisites
1. Clone the code repository by executing git clone in your selected folder:
   ```bash
   https://github.com/StevDoms/SDGE_NAM_Analysis.git
   ```
3. Ensure that [Anaconda](https://www.anaconda.com/products/distribution) is installed.
4. Create a `data/raw` directory and place the following data within:
    ```
    data/raw/
        gis_weatherstation_shape_2024_10_04.csv
        src_wings_meteorology_windspeed_snapshot_2023_08_02.csv
        src_vri_snapshot_2024_03_20.csv
        San_Diego_County_Boundary.geojson
        Congressional.shp
        Congressional.shx
    ```
---

### Step 1: Setting up the Conda Environment

1. Open a terminal and navigate to the project directory:
   ```bash
   cd SDGE_NAM_Analysis
   ```

2. Run the following commands to set up the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate venv
   ```
   Note: if you are running the code in DSMLP, you might get the following error `CondaError: Run 'conda init' before 'conda activate'`.   
   If this happens, run the following:   

   ```bash
   conda init
   source ~/.bashrc
   conda activate venv
   ```

3. After the Conda environment is activated, the project is ready to run.

---

### Step 2: Model Training  

This step involves training the LightGBM model by obtaining necessary datasets, preprocessing data, and running the model.  

#### 1. Obtain NAM Data  
This script scrapes the SDG&E data website and compiles the NAM data.  
**Note:** It generates a CSV file and only needs to be run once. This script might not work on DSMLP; please run it locally and upload the `nam.csv` into the `data/raw` directory.   
  ```bash
  python run.py create_nam_file
  ```

#### 2. Retrieve Elevation Data
This script gathers elevation data using api.open-elevation.com.   
**Note:** Ensure the **Obtain NAM Data** script has been run first. This script also generates a CSV file and only needs to be run once.
  ```bash
  python run.py create_elevation_file
  ```

#### 3. Preprocess Data
This step processes the collected data to prepare it for training the LightGBM model.
  ```bash
  python run.py process_model_input
  ```

#### 4. Train the LightGBM Model
This script trains the LightGBM model using the preprocessed dataset.
  ```bash
  python run.py light_gbm_model
  ```

#### 5. Make Predictions
This script uses the trained LightGBM model to predict errors on both training and unseen data.
  ```bash
  python run.py predict_model
  ```

#### Running the Full Training Pipeline
To execute all steps in sequence, run:
  ```bash
  python run.py create_nam_file create_elevation_file process_model_input, light_gbm_model predict_model
  ```

#### Recommended Run
For better organization, execute these commands separately as follows:
  ```bash
  python run.py create_nam_file 
  python run.py create_elevation_file 
  python run.py process_model_input, light_gbm_model predict_model
  ```
---

### Step 4: Outputs
Note: Our project's expected outputs are not displayed in the repository as there is a confidentiality agreement with SDG&E. Feel free to run the commands in the terminal or run the proj1_notebook.ipynb to view the output. The notebook will provide a better experience and complete picture of the project.