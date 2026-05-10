# ---------------------------------------------------------
# Helper visualisation functions
# ---------------------------------------------------------
# These small utilities make the workflow cleaner and more readable.
# They handle common tasks:
#   - 'plot_late_rate_by_category()'    <- plotting distributions and category-level late rates
#   - 'plot_numeric_distribution()'     <- bar-chart distribution of numeric values
#   - 'plot_histogram()                 <- simple histogram for numeric variables
#   - 'plot_correlation_matrix()'       <- plotting correlation matrices (numeric + categorical)
#   - 'plot_categorical_association()'  <- heatmap of Cramer's V across categorical columns
#   - 'plot_interactive_pr_curve()'     <- interactice precision recall curve with hover labels
# ---------------------------------------------------------

# ----- Library Imports  -----
# Core data handling
import pandas as pd                  # Data handling for plotting inputs

# Plotting libraries
import matplotlib.pyplot as plt      # Core plotting library
import matplotlib.ticker as mtick    # Tick formatting (commas, intervals)
import seaborn as sns                # High-level statistical plots
import plotly.graph_objects as go    # Interactive plotting 

# Statistical helpers
from scipy import stats as ss        # Required for Cramér’s V heatmap
from sklearn.metrics import precision_recall_curve, auc  # Precision recall curve plotting and AUC calculations.

# Local project utilities
from .utils import cramers_v         # Statistical association measure used in plots
# ---------------------------------------------------------


def plot_late_rate_by_category_grid(df, category_cols, target_col="overdue", ncols=4, 
                                    figsize=(20, 12), hide_index=True):
    """
    Groups by a categorical column and calculates the late‑filing rate.
    Plots a horizontal bar charts for multiple categorical variables in a grid of subplots.
    Returns a dictionary of summary tables.
    """
    n_vars = len(category_cols)                              # Set the number of variables as the length of the  
                                                             # list of category columns noting that binary 
                                                             # columns are treated as category_cols.

    nrows = (n_vars + ncols - 1) // ncols                    # Calculate the number of rows in the grid.

    # Use matplotlib subplot function to define the grid for plotting the charts.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() # flatten for easier indexing

    summaries = {}   # Store the summary tables for later use
    
    # loop over the categorical columns,
    for i, col in enumerate(category_cols):
        summary = (                                              # Create summary dataframe
            df.groupby(col, observed=True)[target_col]           # Group by the selected column
            .mean()                                              # Compute late‑filing rate per category
            .reset_index()                                       # Reset the index to start at zero
            .sort_values(target_col, ascending=False)            # Sort categories by late rate
        )
        
        summaries[col] = summary     # Store the summary tables
        ax = axes[i]                 # Select subplot axis

        # If no bars can be drawn, skip the plot
        if summary.empty:
            ax.set_title(f"{col} (no data in training set)")
            ax.axis("off")
            continue

        # Draw barplot
        sns.barplot(                 # Use sns.barplot
            data=summary,            # Set the summary dataframe as the data to use 
            y=col,                   # Define the y variable as the user inputted category column
            x=target_col,            # Define the x variable as the user inputted target column
            orient="h",              # Horizontal bars for readability
            ax=ax                    # Draw on the correct subplot axis
        )

        # Only label bars if they exist
        if ax.containers:
            # Add labels to bars
            ax.bar_label(
                ax.containers[0],
                fmt="%.2f",              # Show late rate on each bar
                label_type="center",     # Centre the labels
                color="white"            # Make the text white
            )
        
        # Set the titles and labels to use on the plot.
        ax.set_title(f"Late Filing Rate By {col.replace('_', ' ').title()}") # Replace underscores with spaces
                                                                             # in column names for title names.
        ax.set_xlabel("Proportion Late")  # x_axis label as proportion late
        ax.set_ylabel(col.replace('_', ' ').title())  # Use column names as y-axis labels replacing underscores.

    # Remove unused subplots if any exist
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()   # Tighten the plots for space saving, avoids clipped labels
    plt.show()           # Display the bar chart

    return None          # Do not return tables
    

def plot_numeric_distribution(df, column, figsize=(10, 5), tick_interval=None):
    """
    Creates a bar chart showing the distribution of a numeric variable.
    Supports custom tick spacing and returns the underlying distribution.
    """
    # Create a table of value counts sorted by the numeric value
    dist = (
        df[column]
        .value_counts()
        .sort_index()                               # Keep values in numeric order
        .reset_index(name="count")                  # Reset the index and name the column "count"
        .rename(columns={"index": column})
    )
    
    plt.figure(figsize=figsize)                     # Define the plot figure size
    
    # Draw the bar chart
    ax = sns.barplot(     # Use the seaborn library barplot function
        data=dist,        # Define the table to use 'dist' from above
        x=column,         # Define the column to plot (user defines this when calling the function.)
        y="count"         # Use the 'count' column for y-values
    )
    
    # Optional: control spacing between x‑axis ticks
    if tick_interval is not None:
        ax.xaxis.set_major_locator(mtick.MultipleLocator(tick_interval))
    
    # Format y‑axis with commas and avoid scientific notation
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    # Set the titles and labels to use on the plot.
    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel("Number of Companies")
    
    plt.tight_layout()   # Tighten the plot for space saving, avoid clipped labels
    plt.show()           # Display the 
    
    return dist          # Return the underlying distribution table for further inspection or reuse


def plot_histogram(df, column, bins=40, figsize=(10, 5)):
    """
    Simple histogram for numeric variables.
    Removes scientific notation and formats counts cleanly.
    """
    plt.figure(figsize=figsize)          # Define the size of the plot on the page
    
    df[column].dropna().hist(bins=bins)  # Ensure the user defined column is used, removing NaN values,
                                         # and using number of bins defined by the user, default is 40.

    # Set the titles and labels to use on the plot.
    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel("Frequency")
    
    ax = plt.gca()     # Get current axis
    
    # Remove scientific notation + add commas
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    
    plt.tight_layout()   # Tighten the plot for space saving, avoid clipped labels
    plt.show()           # Display the histogram


def plot_correlation_matrix(df, figsize=(8, 6), annot=True, fmt=".2f", cmap="coolwarm", return_matrix=False):
    """
    Computes and plots a correlation matrix for numeric variables.
    Useful for spotting multicollinearity or strong relationships.
    """
    numeric_df = df.select_dtypes(include="number")     # Keep numeric columns only
    corr_matrix = numeric_df.corr()                     # Compute pairwise correlations
    
    plt.figure(figsize=figsize)     # Set the plot area figure size
    
    sns.heatmap(                    # Use seaborns heatmap function
        corr_matrix,                # Define which datafram to use
        annot=annot,                # Show correlation values
        fmt=fmt,                    # Number formatting
        cmap=cmap,                  # Colour palette
        linewidths=0.5              # Thin line grids
    )
    
    plt.title("Correlation Matrix for Numeric Variables")  # Set the title
    plt.tight_layout()   # Tighten the plot for space saving, avoid clipped labels
    plt.show()           # Display the heatmap
    
    if return_matrix:
        return corr_matrix    # Return the matrix for further analysis


def plot_categorical_association(df, figsize=(10, 8), annot=True):
    """
    Computes pairwise Cramers V for all categorical columns.
    Plots a heatmap showing categorical associations.
    Helps identify redundant or highly related categorical features.
    """
    # Select categorical columns only
    cat_df = df.select_dtypes(include=["object", "category", "bool", "int8"])

    # Needs to have at least two categorical variable to comput the pairwise associations
    if cat_df.shape[1] < 2:
        raise ValueError("Need at least two categorical columns")

    cols = cat_df.columns
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float) # Empty results matrix to start with

    # Compute pairwise Cramers V for each column pair for the matrix
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
            else:
                matrix.loc[col1, col2] = cramers_v(cat_df[col1], cat_df[col2])

    # Plot the heatmap
    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=annot,      # Show values in cells
        fmt=".2f",        # Format the numbers
        cmap="coolwarm",  # Colour palette
        linewidths=0.5    # Thin gridlines
    )

    plt.title("Categorical Association Matrix (Cramér's V)")    # Set the title
    plt.tight_layout()   # Tighten the plot for space saving, avoid clipped labels
    plt.show()           # Display the heatmap

    return matrix        # Return the full association matrix for further analysis


def plot_interactive_pr_curve(y_true, y_prob, title, save_prefix):
    # compute precision, recall, and thresholds for the model
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # calculate PR-AUC for labelling
    pr_auc = auc(recall, precision)

    # build interactive precision–recall curve
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines+markers',
        name='Precision–Recall Curve',
        hovertemplate='Recall: %{x:.4f}<br>Precision: %{y:.6f}<extra></extra>'
    ))

    # configure layout for clarity
    fig.update_layout(
        title=f"{title}<br><sup>PR_AUC = {pr_auc:.6f}</sup>",
        xaxis_title="Recall",
        yaxis_title="Precision",
        hovermode='closest',
        width=1000, height=700,
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05)  # optional zoom slider
    ))


    # save interactive HTML for GitHub
    fig.write_html(f"{save_prefix}.html")

    # save static PNG for reports
    fig.write_image(f"{save_prefix}.png", scale=3)

    # display interactive chart in notebook
    fig.show()