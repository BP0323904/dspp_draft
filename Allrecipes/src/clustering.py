# ----- Library Imports  -----

# Core libraries
import pandas as pd                               # DataFrame manipulation and tabular data handling
import matplotlib.pyplot as plt                   # Static plotting
import seaborn as sns                             # Statistical visualisation

# Machine learning (K-Means clustering)
from sklearn.preprocessing import StandardScaler  # Feature scaling before clustering
from sklearn.cluster import KMeans                # K-Means clustering algorithm
from sklearn.metrics import silhouette_score      # Silhouette score for assessing the fitness of the clustering model
from sklearn.decomposition import PCA             # Principal Component Analysis graph plotting



# ---- Helper Functions -----

# K-Means helper function for scaling the columns in the data manually before fitting the final model, ensuring the mean = 0 and st.dev = 1.
# This function uses scikit‑learn’s StandardScaler to standardise the numeric features, returning both the scaled NumPy array and the fitted scaler object.
def scale_features(df):
    """
    Scales the numeric features once and returns X_scaled and the scaler.
    """
    scaler = StandardScaler()             # Create a new StandardScaler instance (fresh for each call)
    
    X_scaled = scaler.fit_transform(df)   # Fit the scaler on the raw numeric features and transform them.
                                          # This produces a NumPy array with mean=0 and st.dev=1 per column.
    
    return X_scaled, scaler               # Return both the scaled matrix and the scaler object

# To use it follow this pattern
# X_scaled, scaler = scale_features(df_kmeans)

    
# K-means helper function for plotting the k-means Elbow plot, callable using 'plot_kmeans_inertia()'
# This function uses scikit‑learn’s KMeans to compute inertia values across different cluster counts and matplotlib.pyplot to plot the resulting elbow curve.
def plot_kmeans_inertia(X_scaled, k_min=1, k_max=20, random_state=42):
    """
    Plots the K-means inertia (within-cluster sum of squares) for a range of k values.
    """
    inertia = []     # Create an empty list to store the inertia values.
    ks = range(k_min, k_max + 1)  # Set the range (inclusive of k_max)

    for k in ks:
        # Fit K-Means for each k, fitting on the scaled data, finding the best cluster centres for k.
        # Random state ensures reproducibility.
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X_scaled)

        # Store the inertia
        inertia.append(kmeans.inertia_)

    # Plot the inertia vs k elbow plot.
    plt.figure(figsize=(7, 4))
    plt.plot(ks, inertia, marker='o')  # Marker shape for each k
    plt.xlabel("Number of clusters (k)")  # x-axis label
    plt.ylabel("Inertia")                 # y-axis label
    plt.title("K-means Elbow Plot")       # Title
    plt.grid(True)                        # Enable the background grid
    plt.show()                            # Display the graph/plot


# K-means helper function for plotting the k-means silhouette plot, callable using 'plot_kmeans_silhouette()'
# This function uses scikit‑learn’s KMeans and silhouette_score to compute silhouette values across different cluster counts, and matplotlib.pyplot to plot the resulting silhouette curve.
def plot_kmeans_silhouette(X_scaled, k_min=2, k_max=20, random_state=42):
    """
    Plots the K-means silhouette score for a range of k values.
    k_min = 2, the minimum number of clusters to use.
    k_max = n, the maximum number of clusters to use.
    Example k_min=2 and k_max=10 will run from 2 to 10+1=11,
    which will give 10 iterations (2clusters, 3clusters, 4, 5, 6, 7, 8, 9, 10clusters, 11clusters).
    """
    silhouette_scores = []     # Create an empty list to store the silhouette values.
    ks = range(k_min, k_max + 1)  # Set the range (inclusive of k_max)

    for k in ks:
        # Fit K-Means for each k and obtain cluster labels.
        # Random state ensures reproducibility.
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X_scaled)

        # Compute and store the silhouette score
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    # Plot the silhouette score vs k
    plt.figure(figsize=(7, 4))
    plt.plot(ks, silhouette_scores, marker='o')  # Marker shape for each k
    plt.xlabel("Number of clusters (k)")         # x-axis label
    plt.ylabel("Silhouette Score")               # y-axis label
    plt.title("K-means Silhouette Analysis")     # Title
    plt.grid(True)                               # Enable the background grid
    plt.show()                                   # Display the graph/plot


# K-Means fit helper function for fitting the model and assigning labels, callable using 'fit_means_and_assign_labels'
# This helper function scales the data, fits k-means, returns the labels, assigns them back to the dataframe
# returns the fitted model.
def fit_kmeans_and_assign_labels(X_scaled, n_clusters=5, random_state=42):
    """
    Fits K-means on pre-scaled data features and assigns the resulting
    cluster labels back into the SAME DataFrame.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


# K-Means PCA 2D helper function for plotting the clusters on 2D axes, callable using 'plot_pca_2d'
# This helper function plots a graph which is used to help decide upon the number of clusters.
# This function uses scikit‑learn’s PCA to compute the two principal components, pandas to assemble the transformed data into a DataFrame, and seaborn/matplotlib to plot the 2D cluster visualisation with centroid markers.
def plot_pca_2d(X_scaled, cluster_labels):
    """
    Performs PCA (2 components) on the scaled feature matrix and plots
    a 2D scatter plot coloured by cluster labels, including centroid markers.

    Parameters
    ----------
    X_scaled : array-like
        Scaled feature matrix used for clustering.
    cluster_labels : array-like
        Cluster labels produced by the chosen K-means model.
    """

    # 1. Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 2. Build DataFrame
    df_temp = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": cluster_labels
    })

    # Convert numeric labels → "Cluster X"
    df_temp["cluster_label"] = df_temp["cluster"].apply(lambda x: f"Cluster {x}")

    # 3. Compute centroids
    centroids = df_temp.groupby("cluster_label")[["PC1", "PC2"]].mean().reset_index()

    # 4. Plot
    plt.figure(figsize=(7, 4))
    sns.scatterplot(
        data=df_temp,
        x="PC1", y="PC2",
        hue="cluster_label",
        palette="tab10",
        s=50
    )

    # 5. Add centroid markers (no text)
    plt.scatter(
        centroids["PC1"],
        centroids["PC2"],
        c="yellow",
        s=100,
        marker="X",
        edgecolor="black",
        linewidth=1.2,
        label="Centroids"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Visualisation of K-means Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# K-Means PCA 3D helper function for plotting the clusters on 3D axes, callable using 'plot_pca_3d'
# This helper function plots a graph which is used to help decide upon the number of clusters.
# This function uses scikit‑learn’s PCA to compute the three principal components, pandas to organise the transformed data, and seaborn/matplotlib (including matplotlib’s 3D projection) to produce the 3D cluster visualisation with centroid markers.
def plot_pca_3d(X_scaled, cluster_labels):
    """
    Performs PCA (3 components) on the scaled feature matrix and plots
    a 3D scatter plot coloured by cluster labels, including centroid markers.

    Parameters
    ----------
    X_scaled : array-like
        Scaled feature matrix used for clustering.
    cluster_labels : array-like
        Cluster labels produced by the chosen K-means model.
    """

    # 1. Fit PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # 2. Build DataFrame
    df_temp = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "PC3": X_pca[:, 2],
        "cluster": cluster_labels
    })

    # Convert numeric labels → "Cluster X"
    df_temp["cluster_label"] = df_temp["cluster"].apply(lambda x: f"Cluster {x}")

    # 3. Compute centroids
    centroids = df_temp.groupby("cluster_label")[["PC1", "PC2", "PC3"]].mean().reset_index()

    # 4. Prepare colours (consistent with 2D plot)
    unique_labels = sorted(df_temp["cluster_label"].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    # 5. Plot
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster separately for a clean legend
    for label in unique_labels:
        subset = df_temp[df_temp["cluster_label"] == label]
        ax.scatter(
            subset["PC1"], subset["PC2"], subset["PC3"],
            s=50,
            color=color_map[label],
            label=label,
            depthshade=False   # prevents shading from hiding points
        )

    # 6. Add centroid markers LAST so they appear on top
    ax.scatter(
        centroids["PC1"], centroids["PC2"], centroids["PC3"],
        c="yellow",
        s=100,
        marker="X",
        edgecolor="black",
        linewidth=1.2,
        depthshade=False,     # ensures visibility
        label="Centroids"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D Visualisation of K-means Clusters")

    # Move legend outside the plot
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()