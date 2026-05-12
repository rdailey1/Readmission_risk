from preprocessing import preprocess_data
from model_training import train_models, evaluate_models

processed_df = preprocess_data()

logistic_model, random_forest_model, X_train, X_test, y_train, y_test = train_models(
    processed_df
)
evaluate_models(
    logistic_model,
    random_forest_model,
    X_test,
    y_test
)

print("Preprocessing, training, and evaluation complete from main.")