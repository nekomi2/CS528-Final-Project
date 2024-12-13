import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import joblib
import os
from tqdm import tqdm

lookup = {"Up": 0, "Down": 1, "Left": 2, "Right": 3, "Neutral": 4, "Clap": 5}


def load_data(data_dir, window_size=800):
    features = []
    labels = []

    for label_name in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        if label_name not in lookup:
            print(f"Warning: Label '{label_name}' not in lookup. Skipping.")
            continue

        label = lookup[label_name]

        for filename in tqdm(os.listdir(label_path), desc=f"Loading {label_name}"):
            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(label_path, filename)
            try:
                df = pd.read_csv(file_path)

                data = (
                    df[["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]]
                    .values[:window_size]
                    .astype(np.float32)
                )

                data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
                features.append(data.flatten())
                labels.append(label)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def train_and_evaluate_svm(
    features, labels, n_splits=5, model_path="svm_model.joblib", random_state=69420
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracies = []
    all_conf_matrix = np.zeros((len(lookup), len(lookup)), dtype=int)
    detailed_results = pd.DataFrame(
        columns=["Fold", "True Label", "Predicted Label", "Confidence Score"]
    )

    fold_num = 1
    for train_index, test_index in skf.split(features, labels):
        print(f"\n=== Fold {fold_num} ===")
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        svm_classifier = SVC(kernel="rbf", probability=True, random_state=random_state)
        svm_classifier.fit(X_train, y_train)

        y_pred = svm_classifier.predict(X_test)
        y_probs = svm_classifier.predict_proba(X_test)
        confidence_scores = np.max(y_probs, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"SVM Accuracy for Fold {fold_num}: {accuracy:.2%}")

        conf_matrix = confusion_matrix(y_test, y_pred, labels=list(lookup.values()))
        all_conf_matrix += conf_matrix

        fold_results = pd.DataFrame(
            {
                "Fold": fold_num,
                "True Label": y_test,
                "Predicted Label": y_pred,
                "Confidence Score": confidence_scores,
            }
        )
        detailed_results = pd.concat(
            [detailed_results, fold_results], ignore_index=True
        )

        fold_num += 1

    print("\n=== Cross-Validation Summary ===")
    print(f"Average Accuracy: {np.mean(accuracies):.2%}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        all_conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=lookup.keys(),
        yticklabels=lookup.keys(),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    model = SVC(kernel="rbf", probability=True, random_state=random_state)
    model.fit(features, labels)
    joblib.dump(model, model_path)
    print(f"Final SVM model saved to '{model_path}'.")


if __name__ == "__main__":
    data_directory = "Data/"
    print("Loading data from directory...")
    X, y = load_data(data_directory)
    print(f"Total samples loaded: {len(y)}")
    train_and_evaluate_svm(X, y)
