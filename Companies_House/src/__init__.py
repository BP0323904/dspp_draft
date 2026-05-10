# Re-export key helper functions so they can be imported directly from src.
# This makes notebook imports cleaner and keeps the project structured like a real Python package.

from .utils import (                 # Import selected functions from the utils module
    system_summary,                  # Produces summary of system and libraries used
    find_project_root,               # Locates the project root, for loading and saving data
    mask_user,                       # Mask the username when printing in notebooks
    display_table,                   # displaying tables without an index
    print_dataset_shape,             # reporting dataset shape
    drop_columns_and_report,         # dropping columns with a shape check
    value_counts_table,              # producing value‑count tables with percentages
    missing_values_table,            # summarising missing values
    map_categories,                  # mapping messy category labels to cleaner groups
    cramers_v                        # computing Cramér's V for categorical associations
)

from .visualisations import (        # Import selected functions from the visualisation module
    plot_late_rate_by_category_grid, # plotting distributions and category-level late rates in a grid
    plot_numeric_distribution,       # bar-chart distribution of numeric values
    plot_histogram,                  # simple histogram for numeric variables
    plot_correlation_matrix,         # plotting correlation matrices (numeric + categorical)
    plot_categorical_association,    # heatmap of Cramer's V across categorical columns)
    plot_interactive_pr_curve        # interactive precision recall curve with hover labels
)

#from .logistic_regression import (   # Import selected functions from the logisitc regression module
    
#)

# __all__ defines the public API of the src package.
# Only the names listed here will be imported when using: from src import *
__all__ = [
    "system_summary",                  # System and library summary information helper (utils.py)
    "find_project_root",               # Dataset summary helper (utils.py)
    "mask_user",                       # Masking the username helper (utils.py)
    "display_table",                   # Column cleaning helper (utils.py)
    "print_dataset_shape",             # reporting dataset shape helper (utils.py)
    "drop_columns_and_report",         # dropping columns with a shape check helper (utils.py)
    "value_counts_table",              # producing value‑count tables with percentages helper (utils.py)
    "missing_values_table",            # summarising missing values helper (utils.py)
    "map_categories",                  # mapping messy category labels to cleaner groups helper (utils.py)  
    "cramers_v",                       # computing Cramér's V for categorical associations helper (logistic_regression.py)
    "plot_late_rate_by_category_grid", # plotting distributions and category-level late rates helper (visualisations.py)
    "plot_numeric_distribution",       # bar-chart distribution of numeric values helper (visualisations.py)
    "plot_histogram",                  # simple histogram for numeric variables helper (visualisations.py)
    "plot_correlation_matrix",         # plotting correlation matrices (numeric + categorical) helper (visualisations.py)
    "plot_categorical_association",    # heatmap of Cramer's V across categorical columns) helper (visualisations.py)
    "plot_interactive_pr_curve"        # interactive precision recall curve with hover labels helper (visualisations.py)
]