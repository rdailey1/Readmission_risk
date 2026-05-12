from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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