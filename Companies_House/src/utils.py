# ---------------------------------------------------------
# Helper utility functions
# ---------------------------------------------------------
# These small utilities make the workflow cleaner and more readable.
# They handle common tasks:
#   - 'system_summary()'                <- displays summary information about Python libraries and computer system
#   - 'find_project_root()'             <- Locates the project root, for loading and saving data
#   - 'display_table()'                 <- displaying tables without an index
#   - 'print_dataset_shape()'           <- reporting dataset shape
#   - 'drop_columns_and_report()'       <- dropping columns with a shape check
#   - 'value_counts_table()'            <- producing value‑count tables with percentages
#   - 'missing_values_table()'          <- summarising missing values
#   - 'map_categories()'                <- mapping messy category labels to cleaner groups
#   - 'cramers_v()'                     <- computing Cramér's V for categorical associations
# ---------------------------------------------------------

# ----- Library Imports  -----
# Core data handling
import pandas as pd              # DataFrame operations used across all utilities
import numpy as np               # Numeric operations (e.g., arrays, sqrt)

# System and environment inspection
import platform                  # OS, Python version, CPU info for system_summary()
import psutil                    # Hardware info (RAM, CPU cores) for system_summary()
import importlib                 # Dynamically import libraries to read their versions

# File system utilities
from pathlib import Path         # Used by find_project_root() to navigate directories

# Statistical helpers
from scipy import stats as ss    # Chi-square test required for cramers_v()

# Display utilities (for using Jupyter)
from IPython.display import display   # Used by display_table() for clean notebook output
# ---------------------------------------------------------


# System and Python library information, callable using the defined name 'system_summary'
def system_summary(libraries=None):
    """
    Displays basic system and Python environment information, along
    with library versions used in the project.
    """

    info = {
        "Operating System": platform.platform(),
        "Python Version": platform.python_version(),
        "CPU Processor": platform.processor(),
        "CPU Physical Cores": psutil.cpu_count(logical=False),
        "CPU Logical Cores": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2)
    }

    if libraries:
        for lib in libraries:
            module = importlib.import_module(lib)
            info[f"{lib} version"] = module.__version__

    return info


# Helper function: walks up the directory tree to locate the project root.
# The folder that contains 'pyproject.toml' is used as the root.
def find_project_root(start: Path) -> Path:
    for path in [start] + list(start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Project root not found")


# Helper function to mask usernames.
def mask_user(path: str) -> str:
    """
    Mask the Windows username in a file path for display.
    """
    user = Path.home().name
    return path.replace(user, "******")


def display_table(df):
    """
    Displays the DataFrame without the index,
    for useful clean, notebook friendly tables.
    """
    return df.style.hide(axis="index") # Hide index for cleaner display


def print_dataset_shape(df):
    """
    Print the number of rows and columns,
    used as a quick check after filtering or cleaning steps.
    """
    print(f"rows: {len(df):,}")             # Show row rount with commas
    print(f"columns: {len(df.columns):,}")  # Show number of columns


def drop_columns_and_report(df, columns):
    """
    Drop specified columns, immediately printing the new dataset shape.
    Helps track how datasets have changed.
    """
    df = df.drop(columns=columns) # Remove unwanted columns
    print_dataset_shape(df)       # Report new shape
    return df                     # Display the new dataframe


def value_counts_table(df, column, hide_index=True, sort=True, sort_by=None, ascending=False):
    """
    Produces a value-counts table with counts, percentages and optional sorting.
    Returns a nicely formatted table for EDA.
    """
    counts = df[column].value_counts().reset_index()                    # Count unique values
    counts.columns = [column, "count"]                                  # Rename columns
    counts["percentage"] = counts["count"] / counts["count"].sum()      # Add percentage column
    
    # Determine sort column
    if sort:
        sort_column = sort_by if sort_by is not None else "count"       # Choose sort column
        counts = counts.sort_values(sort_column, ascending=ascending)
    
    counts["count"] = counts["count"].map("{:,}".format)                # Format counts
    counts["percentage"] = counts["percentage"].map("{:.2%}".format)    # Format percentages
    
    if hide_index:
        return counts.style.hide(axis="index")                          # Return clean, index-free table
    
    return counts   # Display the table


def missing_values_table(df):
    """
    Summarises missing values per column,
    includes counts and percentages, optional sorting.
    """
    result = (
        df.isnull()
        .sum()                                                           # Count missing values per column
        .reset_index()                                                   # Convert to a tidy table
        .rename(columns={"index": "column", 0: "missing"})
    )
    
    result["percentage"] = (result["missing"] / len(df)) * 100           # Missing % of total rows
    
    result = result.sort_values("missing", ascending=False)              # Sort by missing count
    
    result["missing"] = result["missing"].map("{:,}".format)            # Format counts
    result["percentage"] = result["percentage"].map("{:.2f}%".format)   # Format percentages
    
    return result.style.hide(axis="index")                              # Return clean, index‑free table



def map_categories(series, mapping, default=None):
    """
    Replaces category labels using a mapping directory.
    Optional default value for unmapped categories.
    Useful for grouping or simplifying messy categories.
    """
    series = series.astype("string")      # Convert to string so replace is safe and future-proof
    mapped = series.replace(mapping)      # Apply the mapping to the series
    
    if default is not None:               # Apply default if needed
        mapped = mapped.fillna(default)   # Fill any unmapped values with the default value when it is not none.
    return mapped.astype("category")      # Return the cleaned / grouped category series 
                                          # Convert back to category for modelling efficiency


def cramers_v(x, y):
    """
    Computes bias-corrected Cramers V.
    Measures association strangth between two categorical variables.
    """
    confusion = pd.crosstab(x, y)               # Build contingency table
    chi2 = ss.chi2_contingency(confusion)[0]    # Chi-squared statistic
    n = confusion.sum().sum()                   # Total number of observations
    phi2 = chi2 / n                             # Phi-squared (raw association)
    r, k = confusion.shape                      # Table dimensions (rows, columns)

    # Bias correction for small sample sizes or uneven tables
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))  # Return corrected Cramers V.
