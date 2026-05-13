from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def build_models():

    logistic_model = LogisticRegression(
        max_iter=50 #limit optimization iterations to 50
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=20, # start with 20 decision trees to average
        random_state=42
    )

    return logistic_model, random_forest_model


def train_models(processed_df):

    # remove target from features model to train on
    X = processed_df.drop("readmitted", axis=1)
    y = processed_df["readmitted"]

    # split data into test and train 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build models
    logistic_model, random_forest_model = build_models()

    logistic_model.fit(X_train, y_train)
    print("Trained Logistic Regression model")

    random_forest_model.fit(X_train, y_train)
    print("Trained Random Forest model")

    return logistic_model, random_forest_model, X_train, X_test, y_train, y_test


def evaluate_models(logistic_model, random_forest_model, X_test, y_test):

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }

    for model_name, model in models.items():

        predictions = model.predict(X_test)

        # % of correct predictions
        accuracy = accuracy_score(y_test, predictions)
        # how many of predicted + were + (true positives)
        precision = precision_score(y_test, predictions)
        # of all +, how many were found
        recall = recall_score(y_test, predictions)
        # balance of precision and recall
        f1 = f1_score(y_test, predictions)

        print(f"\n{model_name} Evaluation")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


def perform_cross_validation(logistic_model, random_forest_model, X_train, y_train):

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }

    for model_name, model in models.items():

        print(f"\nPerforming cross-validation for {model_name}")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5, # for k-fold cross-validation, data is chunked into 5 (4 folds)
            scoring="f1"
        )

        print("Cross-validation f1 scores:", scores)
        # average f1 across the chunks averaged to improve the fit
        print("Average F1 score:", scores.mean())        


def tune_models(X_train, y_train):

    print("\nStarting Logistic Regression hyperparameter tuning")

    logistic_param_grid = {
        "max_iter": [100, 500],
        "C": [0.1, 1.0],
        "class_weight": [None, "balanced"]
    }

    logistic_grid = GridSearchCV(
        LogisticRegression(),
        logistic_param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    logistic_grid.fit(X_train, y_train)

    print("Best Logistic Regression parameters:", logistic_grid.best_params_)
    print("Best Logistic Regression F1 score:", logistic_grid.best_score_)

    print("\nStarting Random Forest hyperparameter tuning")

    random_forest_param_grid = {
        "n_estimators": [20, 50],
        "max_depth": [None, 10],
        "class_weight": [None, "balanced"]
    }

    random_forest_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        random_forest_param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    random_forest_grid.fit(X_train, y_train)

    print("Best Random Forest parameters:", random_forest_grid.best_params_)
    print("Best Random Forest F1 score:", random_forest_grid.best_score_)

    return logistic_grid.best_estimator_, random_forest_grid.best_estimator_        