import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

def plot_corr(data, method):
    sns.heatmap(data.select_dtypes(include="number").corr(method=method), annot=True, cmap="RdYlGn")
    plt.show()

def saving_pipeline(data, save_folder, save_filename, test_size, random_state):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=data["target"], shuffle=True, random_state=random_state)

    original_save_path = os.path.join(save_folder, f"{save_filename}_original.csv")
    data.to_csv(original_save_path, index=False)

    train_save_path = os.path.join(save_folder, f"{save_filename}_train.csv")
    train_data.to_csv(train_save_path, index=False)

    test_save_path = os.path.join(save_folder, f"{save_filename}_test.csv")
    test_data.to_csv(test_save_path, index=False)