# Re-export key helper functions so they can be imported directly from src.
# This makes notebook imports cleaner and keeps the project structured like a real Python package.

from .utils import (                 # Import selected functions from the utils module
    headline_figures,                # Summarises dataset structure (rows, columns, missing values, memory usage)
    clean_column_headers,            # Cleans column names using regex (lowercase, underscores, alphanumeric only)
    clean_column_contents,           # Cleans column contents using regex, when it is not a missing values (NaN), returns a lowercase alphabet-only string with collapsed whitespace
    display_html,                    # Allows tables to be displayed with clickable links
    add_unique_id,                   # Adds a monotonic id to the data
    time_columns,                    # Shows the chosen dataframe with only the time columns
    time_columns_with_url,           # Shows the chosen dataframe with only the time columns and the url
    time_columns_extended            # Shows the chosen dataframe with only the time columns and the url and the extended time columns
)

from .visualisation import (                   # Import selected functions from the visualisation module
    plot_hist_and_box,                         # Plots a histogram and boxplot side-by-side for exploratory analysis
    plot_bar_counts,                           # Plots bar chart for the specified column
    plot_interactive_scatter_with_ref_line,    # Plots an interactive scatterplot with red y=x reference line
    plot_cluster_boxplots                      # Plots the boxplots of columns grouped on cluster 
)

from .clustering import (            # Import selected functions from the clustering module
    scale_features,                  # Scales numeric features using StandardScaler (mean=0, std=1)
    plot_kmeans_inertia,             # Plots inertia values across k to help choose number of clusters (Elbow method)
    plot_kmeans_silhouette,          # Plots silhouette scores across k to assess cluster separation
    fit_kmeans_and_assign_labels,    # Fits K-Means and returns both the model and cluster labels
    plot_pca_2d,                     # Performs PCA (2 components) and plots a 2D cluster visualisation
    plot_pca_3d                      # Performs PCA (3 components) and plots a 3D cluster visualisation
)

# __all__ defines the public API of the src package.
# Only the names listed here will be imported when using: from src import *
__all__ = [
    "headline_figures",                       # Dataset summary helper (utils.py)
    "clean_column_headers",                   # Column cleaning helper (utils.py)
    "clean_column_contents",                  # Cleans content cleaning helper (utils.py)
    "display_html",                           # Displays a table with clickable links (utils.py)
    "add_unique_id",                          # Creates a monotonic id (utils.py)
    "plot_hist_and_box",                      # Histogram + boxplot helper (visualisation.py)
    "time_columns",                           # Time columns helper (utils.py)
    "time_columns_with_url",                  # Time columns with url helper (utils.py)
    "time_columns_extended",                  # Time columns with url and time behaviour columns helper (utils.py)
    "plot_interactive_scatter_with_ref_line", # Interactive scatterplot with red y=x reference line (visualisation.py)
    "scale_features",                         # Feature scaling helper (clustering.py)
    "plot_bar_counts",                        # Bar chart helper (visualisation.py)
    "plot_kmeans_inertia",                    # Elbow plot helper (clustering.py)
    "plot_kmeans_silhouette",                 # Silhouette plot helper (clustering.py)
    "fit_kmeans_and_assign_labels",           # K-Means fitting helper (clustering.py)
    "plot_pca_2d",                            # PCA 2D visualisation helper (clustering.py)
    "plot_pca_3d",                            # PCA 3D visualisation helper (clustering.py)
    "plot_cluster_boxplots"                   # Boxplots of columns grouped on cluster (visualisation.py) 
    
]