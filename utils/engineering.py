import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

def plot_corr(data, method):
    """
    Plots a heatmap of the correlation matrix for numerical columns in the DataFrame
    Parameters:
    - data: DataFrame for correlation analysis
    - method: Correlation method to use ('pearson', 'kendall', or 'spearman')
    """
    sns.heatmap(data.select_dtypes(include="number").corr(method=method), annot=True, cmap="RdYlGn")
    plt.show()

def saving_datasets(data, save_folder, save_filename):
    """
    Saves the dataset, and creates the specified folder if it doesn't exist
    Parameters:
    - data: DataFrame containing the original dataset
    - save_folder: Folder path where the datasets will be saved
    - save_filename: Base filename for the saved datasets
    Effect:
    - Creates the specified folder if it doesn't exist
    - Saves the original dataset as CSV file with appropriate filename
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    original_save_path = os.path.join(save_folder, f"{save_filename}.csv")
    data.to_csv(original_save_path, index=False)