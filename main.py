from preprocessing import preprocess_data
from model_training import train_models

processed_df = preprocess_data()

logistic_model, random_forest_model, X_train, X_test, y_train, y_test = train_models(
    processed_df
)
print("Preprocessing and model training complete from main.")