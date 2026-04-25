# ----- Library Imports  -----

# System libraries
import platform          # For displaying operating system, python version used and CPU processor used when running the code
import psutil            # For displaying CPU cores and RAM on computer used when running the code           
import importlib         # For displaying the library versions used when running the code


# Core libraries
import pandas as pd                             # DataFrame manipulation and tabular data handling

# Python's regular expressions module
import re                                       # Python's regular expressions module, for text cleaning and pattern matching

# Notebook display utilities
from IPython.display import HTML                # Enables HTML rendering (e.g., hyperlinks in tables)


# ---- Helper Functions -----

# System and Python library information, callable using the defined name '
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


# Headline figures for datasets, callable by using the defined name 'headline_figures()'.
# This function uses pandas to compute dataset‑level summary statistics and assemble them into a one‑row DataFrame.
def headline_figures(df, name):
    """
    Returns a one-row Dataframe summarising key dataset characteristics.
    Can be run multiple times and appended to see varying states of the same dataset, or to
    compare multiple datasets.
    """
    return pd.DataFrame({
        "dataset": [name],
        "n_rows": [len(df)],
        "n_columns": [df.shape[1]],
        "n_numeric_columns": [df.select_dtypes(include="number").shape[1]],
        "n_object_columns": [df.select_dtypes(include="object").shape[1]],
        "total_missing_values": [df.isna().sum().sum()],
        "rows_with_any_missing": [df.isna().any(axis=1).sum()],
        "columns_with_any_missing": [df.isna().any().sum()],
        "memory_usage_MB": [df.memory_usage(deep=True).sum() / (1024**2)]
    })


# Column header cleaning helper function callable by using the defined name 'clean_column_headers()'.
# This function uses pandas’ vectorised string methods to clean and standardise column names, returning a DataFrame with consistently formatted headers.
def clean_column_headers(df):
    """
    Cleans DataFrame column names by:
    - converting to lowercase
    - replacing non-alphanumeric characters with underscores
    - collapsing multiple underscores
    - stripping leading/trailing underscores

    Notes
    -----
    - Keeps naming consistent across the project.
    - Does not modify the original DataFrame in-place.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

# Column contents cleaning helpr function, callable by using the defined name 'clean_column_contents()'
# This function uses pandas to detect missing values NaN before using Python's re library for regular expression pattern matching to cleanse the contents of columns.
def clean_column_contents(text):
    """
    Cleans raw column text by:
    - converting non‑letters to spaces
    - normalising whitespace
    - returning a lowercase, alphabet‑only string

    Notes
    -----
    - Keeps the transformation explicit and reproducible.
    - Does not modify the original DataFrame in-place.
    - Handles missing values safely by returning an empty string.
    """
    if pd.isna(text):
        return ""                              # avoid errors on NaN values

    text = text.lower()                        # ensure consistent casing
    text = re.sub(r'[^a-z\s]', ' ', text)      # keep only letters + spaces
    text = re.sub(r'\s+', ' ', text).strip()   # collapse whitespace (whitespace, tab, double spaces, leading/trailing blanks etc) into single space.
    return text


# Helper function which allows dataframes to display the url clickable links, callable using 'display_html()'
# This function uses pandas for DataFrame manipulation and IPython’s HTML display utilities to render a sorted or sampled subset of the DataFrame as HTML inside a Jupyter notebook.
def display_html(df, n=5, random=False, seed=None, sort_by=None, ascending=True):
    """
    Displays a DataFrame as HTML with optional random sampling and sorting.

    Parameters
    ----------
    df : pandas.DataFrame, The DataFrame to display.
    n : int, default = 5, Number of rows to display.
    random : bool, default = False, if True returns a random sample of n rows, if False returns the first n rows.
    seed : int or None, default = None, random seed for reproducibility when random = True.
    ascending : bool, default = True, if True returns a sorted dataframe

    Returns
    -------
    HTML object for display in Jupyter.
    """
    df_to_show = df.copy()

    if random:
        df_to_show = df_to_show.sample(n=n, random_state=seed)
    else:
        df_to_show = df_to_show.head(n)

    if sort_by:
        df_to_show = df_to_show.sort_values(sort_by, ascending=ascending)    
    
    return HTML(df_to_show.to_html(escape=False))

# How to use:
# display_html(df)                                                     <- first 5 rows, default
# display_html(df, n=10)                                               <- first 10 rows
# display_html(df, random=True)                                        <- random sample of 5 rows
# display_html(df, n=10, random=True, seed=42)                         <- random sample of 10 rows with a seed
# display_html(df, n=10, random=True, seed=42, sort_by="passive_time") <- random sample of 10 rows sorted by passive_time. The location of the sort affects filtering.


# Helper function for creating a monotonic id for the dataframe, callable using 'add_unique_id()'
# This function uses pandas to reset the index, generate a new identifier column, and reorder the DataFrame so the ID column appears first
def add_unique_id(df, id_col="unique_id"):
    """
    Resets the index and creates a user-defined unique ID column.
    The ID column is always placed first in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame, The DataFrame to modify.
    id_col : str, default "unique_id", Name of the identifier column to create (e.g., "recipe_id", "customer_id").

    Returns
    -------
    pandas.DataFrame, Modified DataFrame with the new ID column placed first.
    """
    df = df.copy()

    # Reset index to ensure it runs from 0...n cleanly, and create the ID column using the index
    df = df.reset_index(drop=True)
    df[id_col] = df.index

    # Reorder columns so the ID column is first
    cols = [id_col] + [c for c in df.columns if c != id_col]

    return df[cols]


# Helper function to gather the information on the time columns within a dataset (for heatmaps, correlations and modelling), callable using 'time_columns()'
# This function uses pandas to select and return only the three numeric time‑related columns from the DataFrame.
def time_columns(df):
    """Return only the numeric time columns."""
    return df[["prep_time", "cook_time", "total_time"]]


# Helper function like the one above but with the url column added in, callable using 'time_columns_with_url()'
# This function uses pandas to select and return the URL column alongside the three numeric time‑related columns for display purposes.
def time_columns_with_url(df):
    """Return URL + time columns for display purposes."""
    return df[["url", "prep_time", "cook_time", "total_time"]]


# Helper function to gather the information on the extended list of time columns within a dataset, callable using 'time_columns_extended()'
# This function uses pandas to select and return the URL column together with all time‑related columns, including the derived active and passive time fields.
def time_columns_extended(df):
    """Return URL + all time columns including active and passive."""
    return df[["url", "prep_time", "cook_time", "total_time", "active_time", "passive_time"]]