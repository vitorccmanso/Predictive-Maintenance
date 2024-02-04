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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, roc_curve, confusion_matrix
from dagshub import dagshub_logger


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


def evaluate_models(X_train, y_train, X_test, y_test, models, params, experiment_name):
    mlflow.set_experiment(experiment_name)
    report = {}
    
    for model_name, model in models.items():
        with mlflow.start_run():
            param = params[model_name]
            
            gs = GridSearchCV(model, param, cv=3, scoring='roc_auc')
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics for classification models
            f1 = f1_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            
            # Log metrics to MLflow
            mlflow.log_params(param)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc_score", roc_auc)
            mlflow.log_metric("recall_score", recall)
            
            # Log the model to DAGsHub
            with dagshub_logger() as logger:
                logger.log_model(model=model, metrics={"f1_score": f1, "roc_auc_score": roc_auc, "recall_score": recall})
            
            # Store the model and predictions for visualization
            report[model_name] = {"f1_score": f1, "roc_auc_score": roc_auc, "recall_score": recall, "model": model, "y_test_pred": y_test_pred}
            
    return report

def visualize_roc_curves(y_test, models):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')

    for model_name, model_data in models.items():
        model = model_data["model"]
        y_test_pred = model_data["y_test_pred"]
        model_roc_auc = roc_auc_score(y_test, y_test_pred)
        fpr, tpr, _ = roc_curve(y_test, y_test_pred)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {model_roc_auc:.3f})')

    plt.legend()
    plt.show()

def visualize_confusion_matrix(y_test, model_name, models):
    plt.figure(figsize=(12, 5))
    
    # Find the best model's parameters logged in MLflow
    best_params = get_best_params_from_mlflow(model_name)
    
    # Get the best model using the found parameters
    best_model = models[model_name]["model"]
    best_model.set_params(**best_params)
    y_test_pred = best_model.predict(X_test)
    
    # Create confusion matrix
    matriz = confusion_matrix(y_test, y_test_pred)
    
    # Plot absolute values
    plt.subplot(1, 2, 1)
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title(f'Confusion Matrix for {model_name} (Best Params)')
    
    # Plot relative values
    plt.subplot(1, 2, 2)
    sns.heatmap(matriz / np.sum(matriz), annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title(f'Normalized Confusion Matrix for {model_name} (Best Params)')
    
    plt.tight_layout()
    plt.show()

def get_best_params_from_mlflow(model_name):
    # Fetch best parameters from MLflow
    run = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(experiment_name).experiment_id, filter_string=f'metrics.model={model_name}').iloc[0]
    best_params = run['params']
    return best_params

# Sample data (replace with your actual data)
X_train, X_test, y_train, y_test = ...

# Modify the experiment name as needed
experiment_name = "Your_Experiment_Name"

def initiate_model_trainer(train_test: tuple):
    X_train, y_train, X_test, y_test = train_test
    
    models = {
        "Logistic Regression": LogisticRegression()
        # "Decision Tree": DecisionTreeClassifier(),
        # "Random Forest": RandomForestClassifier(),
        # "KNN": KNeighborsClassifier()
    }
    
    params = {
        "Logistic Regression": {
            "solver": "liblinear",
            'penalty':['l2', 'l1', None], 
            'C':[1.5,1,0.5,0.01]
        },
        # "Decision Tree": {
        #     'criterion': ['gini', 'entropy'], 
        #     'max_depth': [3, 4, 5, 6, 7, 8], 
        #     'min_samples_split': [1, 2, 3, 4, 5, 6]
        # },
        # "Random Forest":{
        #     'criterion':['gini', 'entropy'],
        #     'max_features':['sqrt','log2',None],
        #     'n_estimators': [8, 16, 32, 64, 128, 256]
        # },
        # "KNN":{
        #     'metric': ['minkowski', 'chebyshev'],
        #     'n_neighbors': [3, 5, 7, 9, 11],
        #     'p': [1, 2, 3, 4]
        # }
    }
    
    model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params, experiment_name=experiment_name)
    
    return model_report