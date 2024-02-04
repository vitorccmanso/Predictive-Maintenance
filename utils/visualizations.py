import matplotlib.pyplot as plt
import seaborn as sns

def create_subplots(rows, columns, figsize=(18,12)):
    """
    Creates a figure and subplots with common settings

    Parameters:
    - rows: Number of rows in the subplot grid
    - columns: Number of columns in the subplot grid
    - figsize: Tuple specifying the width and height of the figure (default is (18, 12))
    
    Returns:
    - fig: The Matplotlib figure object
    - ax: A 1D NumPy array of Matplotlib axes
    """
    fig, ax = plt.subplots(rows, columns, figsize=figsize)
    ax = ax.ravel()
    return fig, ax

def plot_columns(data, cols, plot_func, ax, title_prefix="", x=None):
    """
    Plots specified graphs using the given plotting function

    Parameters:
    - data: DataFrame containing the data to be plotted
    - cols: List of column names to be plotted
    - plot_func: The Seaborn plotting function to be used
    - ax: Matplotlib axes to plot on
    - title_prefix: Prefix to be added to the title of each subplot (default is an empty string)
    - x: Optional parameter for the x-axis when plotting scatter plots (default is None)
    """
    for i, col in enumerate(cols):
        if plot_func == sns.boxplot:
            plot_func(y=data[col], x=data["target"], hue=data["target"], ax=ax[i], legend=False)
        elif plot_func == sns.histplot:
            plot_func(data[col], ax=ax[i], kde=True)
        elif plot_func == sns.scatterplot and x is not None:
            plot_func(y=data[col], x=data[x], hue=data["failure_type"], ax=ax[i])
        else:
            plot_func(x=data[col], ax=ax[i])
        ax[i].set_title(f"{title_prefix}{col.capitalize()}")

def remove_unused_axes(fig, ax, num_plots):
    """
    Removes unused axes from a figure

    Parameters:
    - fig: The Matplotlib figure object
    - ax: A 1D NumPy array of Matplotlib axes
    - num_plots: Number of subplots to keep; remove the rest
    """
    total_axes = len(ax)
    for j in range(num_plots, total_axes):
        fig.delaxes(ax[j])

def numerical_univariate_analysis(data, rows, columns):
    """
    Performs univariate analysis on numerical columns

    Parameters:
    - data: DataFrame containing numerical data for analysis
    - rows: Number of rows in the subplot grid
    - columns: Number of columns in the subplot grid

    Effect:
    - Plots the distribution of numerical columns using seaborn's histplot
    """
    fig, ax = create_subplots(rows, columns)
    cols = data.select_dtypes(include="number")
    plot_columns(data, cols, sns.histplot, ax, title_prefix="Distribution of ")
    remove_unused_axes(fig, ax, cols.shape[1])
    plt.tight_layout()
    plt.show()


def categorical_univariate_analysis(data, rows, columns):
    """
    Performs univariate analysis on categorical columns

    Parameters:
    - data: DataFrame containing categorical data for analysis
    - rows: Number of rows in the subplot grid
    - columns: Number of columns in the subplot grid

    Effect:
    - Plots countplots for each categorical column
    """
    fig, ax = create_subplots(rows, columns)
    cols = data.select_dtypes(include="object")
    plot_columns(data, cols, sns.countplot, ax)
    remove_unused_axes(fig, ax, cols.shape[1])
    plt.tight_layout()
    plt.show()

def features_vs_targets(data, rows, columns):
    """
    Plots numerical features against the target variable

    Parameters:
    - data: DataFrame containing both features and target variable
    - rows: Number of rows in the subplot grid
    - columns: Number of columns in the subplot grid

    Effect:
    - Plots boxplots for numerical features against the target variable
    - Plots a countplot for the target variable against the 'type' column
    """
    fig, ax = create_subplots(rows, columns, figsize=(18, 12))
    cols = data.drop(columns=["target"]).select_dtypes(include="number")
    plot_columns(data, cols, sns.boxplot, ax, "Target x ")
    remove_unused_axes(fig, ax, cols.shape[1])
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(19, 6))
    sns.countplot(data=data, x="type", hue="target")
    plt.title("Target x Type")
    plt.tight_layout()
    plt.show()

def facegrid_hist_target(df, facecol, color):
    """
    Generates FacetGrid histograms for numerical columns based on target values

    Parameters:
    - df: DataFrame containing data for analysis
    - facecol: Column for creating facets in the FacetGrid
    - color: Color for the histograms

    Effect:
    - Creates a FacetGrid for each column based on the unique values in the specified facecol
    - Filters the data to include only rows where the "target" column is equal to 1
    - Shows the resulting FacetGrids with histograms
    """
    for col in df.drop(columns=["target"]).select_dtypes(include="number"):
        g = sns.FacetGrid(df[df["target"]==1], col=facecol)
        g.map(sns.histplot, col, color=color)
        plt.show()

def plot_scatter_numericals_target(data, rows, columns, x):
    """
    Plots scatter plots of numerical columns against x column for target value 1

    Parameters:
    - data: DataFrame containing data for analysis
    - rows: Number of rows in the subplot grid
    - columns: Number of columns in the subplot grid
    - x: Column for the x-axis in scatter plots

    Effect:
    - Plots scatter plots using seaborn's scatterplot for each numerical column against x column
    """
    fig, ax = create_subplots(rows, columns, figsize=(18, 12))
    cols = data.drop(columns=["target", x]).select_dtypes(include='number')
    plot_columns(data[data["target"] == 1], cols, sns.scatterplot, ax, f"{x.capitalize()} x ", x=x)
    remove_unused_axes(fig, ax, cols.shape[1])
    plt.tight_layout()
    plt.show()
