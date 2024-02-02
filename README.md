# Machine Predictive Maintenance Binary Classification

![project header](images/header.png)

## Dataset
The dataset for this project can be found on [Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (licensed under CC0: Public Domain). 

Since real predictive maintenance datasets are generally difficult to obtain and in particular difficult to publish, the creators built this dataset on synthetic data that reflects real predictive maintenance encountered in the industry to the best of their knowledge. The dataset consists of 10000 rows with 10 columns:

- **UID**: Unique identifier ranging from 1 to 10000
- **Product ID**: Consisting of a letter L, M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number
- **Air temperature [K]**: Generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
- **Process temperature [K]**: Generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
- **Rotational speed [rpm]**: Calculated from powepower of 2860 W, overlaid with a normally distributed noise
- **Torque [Nm]**: Torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
- **Tool wear [min]**: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process and a 'machine failure' label that indicates whether the machine has failed in this particular data point for any of the following failure modes are true.
There are two targets present in this dataset. One for binary classification ("Target") and the other for multiclass classification ("Failure Type").
- **Target**: Failure or Not
- **Failure Type**: Type of Failure

## Objectives
The main objective of this project is:

> **To develop a system that will be able to detect what machines will need maintenance before they present a failure**

To achieve this objective, it was further broken down into the following technical sub-objectives:

1. To perform in-depth exploratory data analysis of the dataset
2. To engineer new predictive features from the available features
3. To develop a supervised model to classify behaviour into no failure and failure
4. To create an API endpoint for the trained model and deploy it

## Main Insights
In progress
