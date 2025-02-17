import numpy as np
import pandas as pd
import geopandas as gpd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.cluster import KMeans

def light_gbm(input_gdf: gpd.GeoDataFrame) -> lgb.LGBMRegressor:
    # Define features and target variable
    features = [
        "nam_wind_speed", "nam_elevation_m", "station_elevation_m",
        "nam_distance_from_station_km", "month", "day_of_year"
    ]

    target = "abs_wind_speed_error"
    
    # Define preprocessing for numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),  # Impute missing values
                ("scaler", StandardScaler())  # Standardize numerical features
            ]), features)
        ]
    )
    
    # Define LightGBM model
    model = lgb.LGBMRegressor(random_state=42, verbose=1)
    
    # Define hyperparameter grid for tuning
    param_dist = {
        "n_estimators": [100, 300, 500, 700],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10, -1],  # -1 means no limit
        "num_leaves": [20, 31, 40, 50],
        "min_child_samples": [5, 10, 20, 30],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
    }
    
    # Define pipeline with preprocessing + model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Extract features (X) and target variable (y)
    X = input_gdf[features]
    y = input_gdf[target]
    
    # Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning using RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,  # Number of random combinations to try
        scoring="neg_mean_absolute_error",
        cv=5,  # 5-fold cross-validation
        verbose=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)
    
    # Best model from tuning
    best_model = random_search.best_estimator_
    
    # Evaluate model on test data
    y_pred = best_model.predict(X_test)
    
    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best Hyperparameters: {random_search.best_params_}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")

    return best_model


def predict_light_gbm_model(model: lgb.LGBMRegressor, input_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    features = [
        "nam_wind_speed", "nam_elevation_m", "station_elevation_m",
        "nam_distance_from_station_km", "month", "day_of_year"
    ]
    
    predictor_data = input_data[features]
    target_prediction = model.predict(predictor_data)

    input_data['abs_wind_speed_error_pred'] = target_prediction

    return input_data

def cluster_kmeans_nam_data(nam_gdf: gpd.GeoDataFrame, start_range: int, end_range: int, random_state: int = 42):
    """
    Clusters NAM data using KMeans and adds a 'cluster' column to the GeoDataFrame.

    Parameters:
    - nam_gdf (gpd.GeoDataFrame): A GeoDataFrame containing NAM data with a 'geometry' column.
    - n_clusters (int): Number of clusters for KMeans. Default is 20.
    - random_state (int): Random state for reproducibility. Default is 42.

    Returns:
    - gpd.GeoDataFrame: The input GeoDataFrame with an additional 'cluster' column.
    """

    # Ensure 'geometry' column exists and extract coordinates
    nam_gdf["x"] = nam_gdf.apply(lambda row: row['geometry'].x, axis=1)
    nam_gdf["y"] = nam_gdf.apply(lambda row: row['geometry'].y, axis=1)

    # Standardize features (important for clustering)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(nam_gdf[["x", "y", "abs_wind_speed_error_pred"]])

    silhouette_scores = []
    k_values = range(start_range, end_range)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, labels)
        silhouette_scores.append((k, score))  # Store as (k, score) pair
    
    # Sort by silhouette score in descending order
    sorted_scores = sorted(silhouette_scores, key=lambda x: x[1], reverse=True)
    print(f"Best Cluster: {sorted_scores[0]}")

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=sorted_scores[0][0], random_state=random_state, n_init=10)
    nam_gdf["cluster"] = kmeans.fit_predict(df_scaled)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    return nam_gdf, centroids

    
    


