import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import joblib
import os

lookup = {"Up": 0, "Down": 1, "Left": 2, "Right": 3, "Neutral": 4, "Clap": 5}


def load_data(label_df, data_dir):
    features = []
    labels = []

    for _, row in label_df.iterrows():
        label = row["label"]
        filename = os.path.join(data_dir, label, f'{label}_{row["filename"]:02}.csv')

        df = pd.read_csv(filename)
        # Use first 800 samples from each recording
        data = (
            df[["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]]
            .values[:900]
            .astype(np.float32)
        )
        # Normalize data
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        features.append(data.flatten())
        labels.append(lookup[label])
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def train_and_evaluate_svm(
    X_train, y_train, X_test, y_test, model_path="svm_model.joblib"
):
    svm_classifier = SVC(kernel="rbf", probability=True, random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    y_probs = svm_classifier.predict_proba(X_test)
    confidence_scores = np.max(y_probs, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.2%}")

    results = pd.DataFrame(
        {
            "True Label": y_test,
            "Predicted Label": y_pred,
            "Confidence Score": confidence_scores,
        }
    )

    results["Correct Prediction"] = results["True Label"] == results["Predicted Label"]

    print("\nDetailed Prediction Results:")
    print(results.to_string(index=False))

    correct_confidence = results[results["Correct Prediction"]]["Confidence Score"]
    incorrect_confidence = results[~results["Correct Prediction"]]["Confidence Score"]

    print("\nSummary Statistics:")
    print(f"Total Test Samples: {len(results)}")
    print(
        f"Correct Predictions: {results['Correct Prediction'].sum()} ({(results['Correct Prediction'].mean()*100):.2f}%)"
    )
    print(
        f"Incorrect Predictions: {len(results) - results['Correct Prediction'].sum()} ({((1 - results['Correct Prediction'].mean())*100):.2f}%)"
    )
    print(f"Average Confidence (Correct Predictions): {correct_confidence.mean():.2%}")
    print(
        f"Average Confidence (Incorrect Predictions): {incorrect_confidence.mean():.2%}"
    )

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=svm_classifier.classes_,
        yticklabels=svm_classifier.classes_,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(
        correct_confidence,
        color="green",
        label="Correct Predictions",
        kde=True,
        stat="density",
        linewidth=0,
    )
    sns.histplot(
        incorrect_confidence,
        color="red",
        label="Incorrect Predictions",
        kde=True,
        stat="density",
        linewidth=0,
    )
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    results.to_csv("svm_detailed_results.csv", index=False)
    print("\nDetailed results saved to 'svm_detailed_results.csv'.")

    joblib.dump(svm_classifier, model_path)
    print(f"Trained SVM model saved to '{model_path}'.")


if __name__ == "__main__":
    train_labels = pd.read_csv("Data/train.csv")
    val_labels = pd.read_csv("Data/val.csv")
    train_dir = val_dir = "Data/"
    X_train, y_train = load_data(train_labels, train_dir)
    X_test, y_test = load_data(val_labels, val_dir)
    train_and_evaluate_svm(X_train, y_train, X_test, y_test)
