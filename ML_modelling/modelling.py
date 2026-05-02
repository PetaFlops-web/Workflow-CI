import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000")

train_df = pd.read_csv('dataset_preprocessing/train_clean.csv')
X_train = train_df.drop(columns=['Reached.on.Time_Y.N'])
y_train = train_df['Reached.on.Time_Y.N']

test_df = pd.read_csv('dataset_preprocessing/test_clean.csv')
X_test = test_df.drop(columns=['Reached.on.Time_Y.N'])
y_test = test_df['Reached.on.Time_Y.N']

mlflow.set_experiment("E-Commerce Shipping Late Prediction")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="modelling"):

    n_estimators = 100
    max_depth = 10
    min_samples_split = 2

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

    model.fit(X_train, y_train)

    mlflow.sklearn.save_model(model, "saved_model")