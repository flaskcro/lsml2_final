
# Predict future sales by item and shop
## 1. Project Description 

### 1 Design
- Data : https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales
- Algorithm : Light GBM
- Goal : below RMSE 0.9 Model

### 2 Run Instruction
Training (command): 
    python train.py

Training (mlflow and docker)

   mlflow run -e main .

Docker Image Build:
    docker build -t mlflow-docker-futuresales -f Dockerfile .

Run Docker Image:
    mlflow run docker

### 3. Architecture

Light GBM
- Hyperparameters 

|Hyperparameters | Value |
|---|-------|
|metric| rmse  |
|num_leaves| 255   |
|learning_rate| 0.005 |
|feature_fraction| 0.75 |
|bagging_fraction| 0.75 |
|bagging_freq|5|
|force_col_wise|True|

- Metric

| Data        | RMSE |
|-------------|------|
| Validation  | 0.88399   |
| Public Test | 0.87319   |



