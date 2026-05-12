from preprocessing import preprocess_data
from model_training import train_models, evaluate_models,perform_cross_validation

processed_df = preprocess_data()

logistic_model, random_forest_model, X_train, X_test, y_train, y_test = train_models(
    processed_df
)

# all metrics on a scale of 0-1 where 1 is perfect
evaluate_models(
    logistic_model,
    random_forest_model,
    X_test,
    y_test
)

perform_cross_validation(
    logistic_model,
    random_forest_model,
    X_train,
    y_train
)

print("Preprocessing, training, evaluation, and cross-validation complete from main.")