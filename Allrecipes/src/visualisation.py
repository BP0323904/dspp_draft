# ----- Library Imports  -----

# Core plotting libraries
import matplotlib.pyplot as plt               # Static plotting (figure creation, axes, layout)
import plotly.express as px                   # High-level interactive plotting (scatter, bar, etc.)
import plotly.graph_objects as go             # Low-level Plotly API for custom interactive elements

# Supporting utilities
import math                                   # Mathematical helpers (e.g., ceiling for subplot layout)
import seaborn as sns                         # Statistical visualisation (used for boxplots in some helpers)


# ----- Helper Functions  -----

# Histogram and boxplot helper function, callable using 'plot_hist_and_box()'
# This function uses matplotlib.pyplot to create a side‑by‑side histogram and boxplot, with the data supplied as pandas Series extracted from the DataFrame.
def plot_hist_and_box(df, col, bins=500):
    """
    Plot a histogram (left) and boxplot (right) for a single numeric column.
    Produces a 1x2 layout.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))         # Create a figure with 1 row and 2 columns of subplots

    # Histogram
    axes[0].hist(df[col], bins=bins, color="steelblue")     # Plot the distribution with a histogram
    axes[0].set_title(f"{col} histogram")                   # Add a title to the histogram
    axes[0].set_xlabel(col)                                 # Label the x-axis with a column name
    axes[0].set_ylabel("Frequency")                         # Label the y-axis to show frequency counts

    # Boxplot
    axes[1].boxplot(df[col].dropna(), vert=True)            # Plot a boxplot of the column, dropping NaN values to avoid errors
    axes[1].set_title(f"{col} boxplot")                     # Add a title to the boxplot
    axes[1].set_xlabel(col)                                 # Label the x-axis with the column name

    plt.tight_layout()                                      # Adjust spacing so that labels and titles don't overlap
    plt.show()                                              # Display the plots


# Bar chart plotting helper function, callable using 'plot_bar_counts()'
# This function uses matplotlib.pyplot to generate the bar chart and pandas to compute the value counts that supply the data for the plot.
def plot_bar_counts(df, column, tick_step=None, figsize=(12, 4), color="steelblue"):
    """
    Plot a bar chart of value counts for any column.

    Parameters
    ----------
    df : pandas.DataFrame, the dataset containing the column.
    column : str, the column to count and plot.
    tick_step : int or None, default=None, step size for x-axis ticks (useful for numeric columns), if None, ticks follow the data naturally.
    figsize : tuple, default=(12, 4), figure size for the plot.
    color : str, default="steelblue", bar colour.
    """
    counts = df[column].value_counts().sort_index()         # Count values and sort by index (important for numeric columns)
    
    plt.figure(figsize=figsize)                             # Create the figure
    plt.bar(counts.index, counts.values, color=color)       # Draw the bar chart
    plt.title(f"Distribution of {column}")                  # Add title
    plt.xlabel(column)                                      # Add x-axis label
    plt.ylabel("Count")                                     # Add y-axis label

    # Optional tick spacing for numeric columns
    if tick_step is not None:
        try:
            # Ensure the index is numeric before applying tick spacing
            max_val = int(counts.index.max())
            plt.xticks(range(0, max_val + 1, tick_step))
        except Exception:
            # If the index isn't numeric, silently skip tick spacing
            pass

    plt.tight_layout()                                      # Adjust spacing so that labels and titles don't overlap
    plt.show()                                              # Display the plots

# Example uses
#plot_bar_counts(df, "servings", tick_step=10)
#plot_bar_counts(df, "difficulty")
#plot_bar_counts(df, "rating", color="darkorange")


# Interactive scatterplot helper function, callable using 'plot_interactive_scatter_with_ref_line()'
# This function uses Plotly Express and Plotly Graph Objects to create an interactive scatterplot, with pandas supplying the underlying data
def plot_interactive_scatter_with_ref_line(
    df,
    x,
    y,
    hover_cols=None,
    title=None,
    figsize=(1050, 700),
    save_prefix=None
):
    """
    Create an interactive Plotly scatterplot with a 45-degree reference line.

    Parameters
    ----------
    df : pandas.DataFrame, the dataset containing the columns.
    x : str, column name for x-axis.
    y : str, column name for y-axis.
    hover_cols : list of str, optional, columns to show in hover tooltip.
    title : str, optional, plot title.
    figsize : tuple, default=(1050, 700), width and height of the figure.
    save_prefix : str or None, if provided, saves PNG and HTML using this prefix.
    """

    # Build hover dictionary for Plotly (True = include column in tooltip)
    hover_data = {col: True for col in hover_cols} if hover_cols else None

    # Create the base interactive scatterplot, using Plotly Express for simplicity
    fig = px.scatter(
        df,
        x=x,
        y=y,
        hover_data=hover_data, # Hover tooltips controlled by hover_data
        height=figsize[1]      # Set figure height explicitly
    )

    # Compute the maximum axis value for the 45-degree reference line
    # Ensures the line spans the full visible range
    max_val = max(df[x].max(), df[y].max())

    # Add a dashed orange y=x reference line for comparison
    # Useful for assessing agreement between the two variables
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="orange", dash="dash"),
            showlegend=False
        )
    )

    # Update layout:
    # - Set figure size
    # - Centre the title
    # - Add margins for readability
    # - Add an x-axis range slider for interactive zooming
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        margin=dict(l=50, r=40, t=80, b=60),
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
    )

    # Optional saving:
    # - PNG for static export for placing in the jupyter notebook
    # - HTML for fully interactive version fr github
    if save_prefix:
        fig.write_image(f"{save_prefix}.png", scale=3)
        fig.write_html(f"{save_prefix}.html")

    # Display the interactive figure inside the notebook
    fig.show()


# Helper function for creating a group of boxplots for various columns which are grouped on another column, callable using 'plot_cluster_botplots()'
# This function uses seaborn and matplotlib.pyplot to generate multiple cluster‑grouped boxplots arranged in a subplot grid, with pandas providing the underlying data.
def plot_cluster_boxplots(df, feature_cols, cluster_col="cluster", rows=2):
    """
    Plots multiple boxplots (one per feature) grouped by cluster.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the cluster column and feature columns.
    feature_cols : list of str
        List of numeric columns to plot.
    cluster_col : str, default="cluster"
        Column indicating cluster labels.
    rows : int, default=2
        Number of rows in the subplot grid.

    Notes
    -----
    - Automatically calculates the number of columns needed based on
      the number of features and the chosen number of rows.
    - Hides any unused subplot axes for a clean layout.
    - Keeps all logic explicit and reproducible.
    """

    # Subset the DataFrame to only the cluster column + selected features
    df_plot = df[[cluster_col] + feature_cols].copy()

    # Determine subplot layout
    cols_per_row = math.ceil(len(feature_cols) / rows)

    # Create the subplot grid
    fig, axes = plt.subplots(rows, cols_per_row,
                             figsize=(6 * cols_per_row, 4 * rows))
    axes = axes.flatten()  # Flatten to simplify indexing

    # Plot each feature in its own subplot
    for i, col in enumerate(feature_cols):
        sns.boxplot(data=df_plot, x=cluster_col, y=col, ax=axes[i])
        axes[i].set_title(f"{col} by {cluster_col}")

    # Hide any unused axes (e.g., if features < rows*cols)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()