import configparser
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score, recall_score, roc_curve, confusion_matrix, precision_score, accuracy_score, precision_recall_curve, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
from utils.visualizations import create_subplots

config = configparser.ConfigParser()
config.read("../mlflow_config.ini")

mlflow_tracking_username = config['mlflow']['tracking_username']
mlflow_tracking_password = config['mlflow']['tracking_password']

# Set environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_tracking_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_tracking_password

def preprocess_pipeline(data, test_size:float, target_name:str):
    """
    Preprocesses data, performs one-hot encoding on categorical columns,
    splits into train and test sets, and scales numerical columns

    Parameters:
    - data: Input DataFrame
    - test_size: Proportion of data to be used for the test set
    - target_name: Name of the target variable column

    Returns:
    - tuple: (X_train_preprocessed, X_test_preprocessed, y_train, y_test)
    """
    X = data.drop(columns=[target_name])
    y = data[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y, shuffle=True)

    cat_columns = list(X_train.select_dtypes("object").columns)
    num_columns = list(X_train.select_dtypes(exclude=["object"]).columns)
    numerical_transformer = RobustScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(transformers=[("num", numerical_transformer, num_columns), ("cat", categorical_transformer, cat_columns)], verbose_feature_names_out=False)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=preprocessor.get_feature_names_out())
    X_test_preprocessed = preprocessor.transform(X_test)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=preprocessor.get_feature_names_out())

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test

def initiate_model_trainer(train_test: tuple, experiment_name, use_smote=False):
    mlflow.set_tracking_uri("https://dagshub.com/vitorccmanso/Predictive-Maintenance.mlflow")
    X_train, y_train, X_test, y_test = train_test
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }
    
    params = {
        "Logistic Regression": {
            "solver": ["liblinear", "lbfgs"],
            "penalty":["l2", "l1", None], 
            "C":[1.5, 1, 0.5, 0.1]
        },
        "Random Forest":{
            "criterion":["gini", "entropy", "log_loss"],
            "max_features":["sqrt","log2", None],
            "n_estimators": [8, 16, 32, 64, 128],
            "max_depth": [3, 5, 8, 10],
            "min_samples_split": [2, 5, 8],
            "min_samples_leaf": [1, 3, 5]
        },
        "KNN":{
            "metric": ["minkowski", "chebyshev"],
            "n_neighbors": [3, 5, 7],
            "p": [1, 2, 3]
        }
    }
    
    model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params, experiment_name=experiment_name, use_smote=use_smote)
    
    return model_report

def evaluate_models(X_train, y_train, X_test, y_test, models, params, experiment_name, use_smote):
    mlflow.set_experiment(experiment_name)
    report = {}
    if use_smote:
        # Apply SMOTE to the training data
        smote = SMOTE(sampling_strategy="auto")
        X_train, y_train = smote.fit_resample(X_train, y_train)
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            param = params[model_name]
            if model_name != "KNN":
                param["class_weight"] = [None] if use_smote else ["balanced"]
                param["random_state"] = [42]
            gs = GridSearchCV(model, param, cv=5, scoring=["recall", "f1"], refit="recall")
            grid_result = gs.fit(X_train, y_train)
            model = grid_result.best_estimator_
            y_pred = model.predict(X_test)
            mlflow.set_tags({"model_type": f"{model_name}-{experiment_name}", "smote_applied": use_smote})
            # Calculate metrics for classification models
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            roc = roc_curve(y_test, model.predict_proba(X_test)[:,1])
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics to MLflow
            mlflow.log_params(grid_result.best_params_)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc_score", roc_auc)
            mlflow.log_metric("recall_score", recall)
            mlflow.log_metric("precision_score", precision)
            mlflow.log_metric("accuracy_score", accuracy)
            mlflow.sklearn.log_model(model, model_name, registered_model_name=f"{model_name} - {experiment_name}")
            
            # Store the model and predictions for visualization
            report[model_name] = {"model": model, "y_pred": y_pred, "roc_auc_score": roc_auc, "roc_curve": roc}        
    return report

def visualize_roc_curves(models):
    plt.figure(figsize=(12, 6))
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")

    for model_name, model_data in models.items():
        model_roc_auc = model_data["roc_auc_score"]
        fpr, tpr, thresholds = model_data["roc_curve"]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_roc_auc:.3f})")
    plt.legend()
    plt.show()

def visualize_confusion_matrix(y_test, models, rows, columns):
    fig, ax = create_subplots(rows, columns, figsize=(14, 7))
    for i, (model_name, model_data) in enumerate(models.items()):
        y_pred = model_data["y_pred"]
        matrix = confusion_matrix(y_test, y_pred)
        # Plot the first heatmap
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax[i * 2])
        ax[i * 2].set_title(f"Confusion Matrix: {model_name} - Absolute Values")
        ax[i * 2].set_xlabel("Predicted Values")
        ax[i * 2].set_ylabel("Observed values")

        # Plot the second heatmap
        sns.heatmap(matrix / np.sum(matrix), annot=True, fmt=".2%", cmap="Blues", ax=ax[i * 2 + 1])
        ax[i * 2 + 1].set_title(f"Relative Values")
        ax[i * 2 + 1].set_xlabel("Predicted Values")
        ax[i * 2 + 1].set_ylabel("Observed values")
    fig.tight_layout()
    plt.show()


def plot_precision_recall_threshold(y_test, X_test, models, rows, columns):
    fig, ax = create_subplots(rows, columns, figsize=(16, 6))
    for i, (model_name, model_data) in enumerate(models.items()):
        y_pred_prob = model_data["model"].predict_proba(X_test)[:,1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

        # Plot Precision-Recall vs Thresholds for each model
        ax[i].set_title(f"Precision X Recall vs Thresholds - {model_name}")
        ax[i].plot(thresholds, precisions[:-1], "b--", label="Precision")
        ax[i].plot(thresholds, recalls[:-1], "g-", label="Recall")
        ax[i].plot([0.5, 0.5], [0, 1], 'k--')
        ax[i].set_ylabel("Score")
        ax[i].set_xlabel("Threshold")
        ax[i].legend(loc='center left')

        # Annotate precision and recall at 0.5 threshold
        y_pred = model_data["y_pred"]
        metrics = precision_recall_fscore_support(y_test, y_pred)
        precision = metrics[0][1]
        recall = metrics[1][1]
 
        ax[i].plot(0.5, precision, 'or')
        ax[i].annotate(f'{precision:.2f} precision', (0.51, precision))
        ax[i].plot(0.5, recall, 'or')
        ax[i].annotate(f'{recall:.2f} recall', (0.51, recall))
        ax[i].annotate('0.5 threshold', (0.39, -0.04))

    fig.tight_layout()
    plt.show()

def plot_feature_importance(y_test, X_test, models, metric, rows, columns):
    fig, ax = plt.subplots(rows, columns, figsize=(16,6))
    ax = ax.ravel()
    for i, (model_name, model_data) in enumerate(models.items()):
        result = permutation_importance(model_data["model"], X_test, y_test, n_repeats=10, random_state=42, n_jobs=2, scoring=metric)
        sorted_importances_idx = result['importances_mean'].argsort()
        importances = pd.DataFrame(result.importances[sorted_importances_idx].T, columns=X_test.columns[sorted_importances_idx])
        box = importances.plot.box(vert=False, whis=10, ax=ax[i])
        box.set_title(f"Feature Importance - {model_name}")
        box.axvline(x=0, color="k", linestyle="--")
        box.set_xlabel(f"Decay in {metric}")
        box.figure.tight_layout()
    fig.tight_layout()
    plt.show()