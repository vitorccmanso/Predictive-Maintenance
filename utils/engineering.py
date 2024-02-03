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

def saving_pipeline(data, save_folder, save_filename, test_size, random_state):
    """
    Splits the dataset into train and test sets, saves them, and creates the specified folder if it doesn't exist
    Parameters:
    - data: DataFrame containing the original dataset
    - save_folder: Folder path where the datasets will be saved
    - save_filename: Base filename for the saved datasets
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Seed for reproducibility
    Effect:
    - Creates the specified folder if it doesn't exist
    - Saves the original dataset, train set, and test set as CSV files with appropriate filenames
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=data["target"], shuffle=True, random_state=random_state)

    original_save_path = os.path.join(save_folder, f"{save_filename}_original.csv")
    data.to_csv(original_save_path, index=False)

    train_save_path = os.path.join(save_folder, f"{save_filename}_train.csv")
    train_data.to_csv(train_save_path, index=False)

    test_save_path = os.path.join(save_folder, f"{save_filename}_test.csv")
    test_data.to_csv(test_save_path, index=False)