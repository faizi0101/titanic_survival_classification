import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def main():

    # Load dataset
    df = pd.read_csv("data/raw/train.csv")

    # Select features and target
    features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]

    X = df[features]
    y = df["Survived"]

    # Handle missing values
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # Encode categorical features
    X["Sex"] = X["Sex"].map({
        "male": 0,
        "female": 1
    })

    X = pd.get_dummies(
        X,
        columns=["Embarked"],
        drop_first=True
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create and train the model
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 50)
    print("TITANIC SURVIVAL CLASSIFICATION")
    print("=" * 50)

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()