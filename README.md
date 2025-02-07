# Analyzing Wind Speed Error Using the NAM Data

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
   cd /path/to/SDGE_NAM_Analysis
   ```

2. Run the following commands to set up the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate geo_env
   ```
   Note if you are running in DSMLP you might get an instruction CondaError: Run 'conda init' before 'conda activate', if so run before conda activate geo_env:

   ```bash
   conda init
   source ~/.bashrc
   ```

4. After the Conda environment is activated, you are prepared to run the project code.

---

### Step 2: Building the Project using 

If you need to run specific components of the pipeline, use the following commands:

- **Data Processing**: Prepares and merges weather data.
  ```bash
  python run.py merge
  ```

- **PSPS Probabilities**: Calculates PSPS probabilities for weather stations.
  ```bash
  python run.py psps
  ```

- **Filter PSPS Stations**: Filters high-risk PSPS stations based on a threshold.
  ```bash
  python run.py filter
  ```

- **VRI and Conductor Merge**: Merges conductor and vegetation risk index (VRI) data.
  ```bash
  python run.py merge_vri
  ```

- **Analyze spans**: Builds a directed graph of spans for upstream/downstream analysis to perform span analysis and and calculate probabilities of each span.
  ```bash
  python run.py analyze_spans
  ```

- **Feeder analysis**: Perform feeder analysis by exploring the annual customers affected for a given parent feeder id and predicting number of customers affected in 10 years.
  ```bash
  python run.py feeder_analysis
  ```

---

### Step 4: Outputs
Note: Our project's expected outputs are not displayed in the repository as there is a confidentiality agreement with SDG&E. Feel free to run the commands in the terminal or run the proj1_notebook.ipynb to view the output. The notebook will provide a better experience and complete picture of the project.