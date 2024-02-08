import pandas as pd
import mlflow
import os
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

uri = "https://dagshub.com/vitorccmanso/Predictive-Maintenance.mlflow"

class PredictPipeline:
    """
    A class representing a predictive pipeline for making predictions on datasets.
    This class can load the model from MLflow, preprocess the manual data and dataset that the user input
    and make the predictions
    
    Attributes:
    - client (mlflow.MlflowClient): MLflow client for accessing models and runs
    - loaded_model: Loaded machine learning model for making predictions
    """
    def __init__(self):
        """
        Initializes the PredictPipeline object
        """
        mlflow.set_tracking_uri(uri)
        self.client = mlflow.MlflowClient(tracking_uri=uri)
        self.loaded_model = self.load_model_once()

    def load_model_once(self):
        """
        Loads the machine learning model once from the MLflow tracking server

        Returns:
        - loaded_model: The loaded machine learning model
        """
        registered_model = self.client.get_registered_model('Random Forest - All_Features')
        run_id = registered_model.latest_versions[-1].run_id
        logged_model = f'runs:/{run_id}/Random Forest'
        loaded_model = mlflow.sklearn.load_model(logged_model)
        return loaded_model

    def process_dataset(self, input_data):
        """
        Processes the input dataset by filtering and reordering columns

        Parameters:
        - input_data: The input dataset

        Returns:
        - pd.DataFrame: The processed dataset
        
        Raises:
        - ValueError: If the input dataset does not contain all the required columns
        """
        columns = ["type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
        input_data.columns = input_data.columns.str.lower().str.replace(r'\[.*\]', '', regex=True).str.rstrip().str.replace(' ', '_')
        if not set(columns).issubset(input_data.columns):
            raise ValueError("Dataset must contain all the columns listed above")
        filtered_data = input_data[columns]
        reordered_data = filtered_data.reindex(columns=columns)
        return reordered_data

    def preprocess_data(self, data, manual):
        """
        Preprocesses the input data by scaling numerical features and encoding categorical features

        Parameters:
        - data: The input data
        - manual (bool): Indicates whether it is a dataset prediction or a manual data input prediction

        Returns:
        - pd.DataFrame: The preprocessed data
        """
        if manual:
            data_scale = pd.read_csv("app/Scale_Data/data_scale.csv")
            data = pd.concat([data, data_scale], axis=0)
        data['power'] = data['torque'] * data['rotational_speed']
        data['diference_temperature'] = data['air_temperature'] - data['process_temperature']
        nums = data.select_dtypes("number").columns
        cats = data.select_dtypes("object").columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), nums), 
                ('cat', OneHotEncoder(categories=[["H", "L", "M"]]), cats) 
            ], verbose_feature_names_out=False)
        data = preprocessor.fit_transform(data)
        new_data = pd.DataFrame(data, columns=preprocessor.get_feature_names_out())
        if manual:
            return new_data.head(1)
        return new_data

    def predict(self, data, manual=False):
        """
        Makes predictions on the input data

        Parameters:
        - data: The input data
        - manual: Indicates whether it is a dataset prediction or a manual data input prediction

        Returns:
        - Union[str, List[str]]: The predicted class or classes
        """
        preds_proba = self.loaded_model.predict_proba(self.preprocess_data(data, manual))[:,1]
        preds = (preds_proba >= 0.6).astype(int)
        prediction_classes = ["Healthy", "Needs Maintenance"]
        if manual:
            predicted_class = prediction_classes[preds[0]]
            return predicted_class
        predicted_classes = [prediction_classes[pred] for pred in preds]
        return predicted_classes

class CustomData:
    """
    A class representing custom data with specific attributes

    Attributes:
    - type (str): The type of product
    - air_temperature (float): The air temperature
    - process_temperature (float): The process temperature
    - rotational_speed (float): The rotational speed
    - torque (float): The torque
    - tool_wear (float): The tool wear
    """
    def __init__(self, type:str, air_temperature:float, process_temperature:float, rotational_speed:float, torque:float, tool_wear:float):
        """
        Initializes the CustomData object with the provided attributes
        """
        self.type = type
        self.air_temperature = air_temperature
        self.process_temperature = process_temperature
        self.rotational_speed = rotational_speed
        self.torque = torque
        self.tool_wear = tool_wear

    def get_data_as_dataframe(self):
        """
        Converts the CustomData object into a pandas DataFrame

        Returns:
        - pd.DataFrame: The CustomData object as a DataFrame
        """
        custom_data_input_dict = {
            "type": [self.type],
            "air_temperature": [self.air_temperature],
            "process_temperature": [self.process_temperature],
            "rotational_speed": [self.rotational_speed],
            "torque": [self.torque],
            "tool_wear": [self.tool_wear],
        }
        return pd.DataFrame(custom_data_input_dict)