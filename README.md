# TimeSeries-Clustering

This project demonstrates various clustering methods and visualizations for time series data. It includes utility functions for running clustering algorithms, plotting results, and performing silhouette analysis.

## Team Members

- Abdesselam BENAMEUR
- Hakim IGUENI
- Salma TALANTIKITE

## Contents

- `utils.py`: Contains utility functions for clustering and visualization.
- `ProjetUL.ipynb`: Jupyter notebook demonstrating the use of the utility functions on a dataset.

## Setup

### Requirements

Make sure you have the following Python packages installed:

- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- tslearn
- kmedoids

You can install these packages using `pip`:

\`\`\`bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tslearn kmedoids
\`\`\`

### Data

The data used in this project is loaded from the following URLs:

- `X`: [Dataset URL](http://allousame.free.fr/mlds/donnees/X.txt)
- `APPART`: [Dataset URL](http://allousame.free.fr/mlds/donnees/APPART.txt)
- `JOUR`: [Dataset URL](http://allousame.free.fr/mlds/donnees/JOUR.txt)

## Usage

### Utility Functions

The `utils.py` file contains the following utility functions:

- `run_clustering_method(clustering_method, X, metric='precomputed', verbose=True)`: Runs a clustering algorithm for a range of cluster numbers (2 to 10), computes silhouette scores and inertia, and prints the results.
- `plot_silhouette_analysis(X, labels, title)`: Plots the silhouette analysis for the clustering results.
- `plot_heatmap(Y, cmap=[green, red, blue, yellow])`: Plots a heatmap of the given matrix `Y`.
- `visualize_clusters(X, labels, title)`: Visualizes clusters by plotting time series data that belong to the same cluster.

### Jupyter Notebook

The `ProjetUL.ipynb` notebook demonstrates the use of the utility functions on a dataset. It includes the following steps:

1. **Read the data**: Load the dataset from the provided URLs.
2. **Calculate the distance matrix**: Compute the distance matrix using Euclidean and DTW distances.
3. **Clustering**: Apply KMedoids and KMeans clustering methods.
4. **Visualization**: Use elbow method, silhouette scores, heatmaps, and cluster visualizations to analyze the results.

#### Example Code

\`\`\`python
# Import utility functions
from utils import *

# Load the data
X = pd.read_csv("http://allousame.free.fr/mlds/donnees/X.txt", sep=" ", header=None)

# Calculate distance matrix using Euclidean distance
from scipy.spatial.distance import pdist, squareform
dist_matrix_euc = pdist(X, metric='euclidean')
dist_matrix_euc = squareform(dist_matrix_euc)

# Run KMeans clustering
from sklearn.cluster import KMeans
clust_method = lambda k: KMeans(n_clusters=k, random_state=0, n_init='auto')
inertias, silhouettes, labels = run_clustering_method(clust_method, X, metric="euclidean")

# Plot results
plot_inertia(inertias, title="Elbow method for KMeans with Euclidean distance")
plot_silhouette(silhouettes, title="Silhouette score for KMeans with Euclidean distance")
visualize_clusters(X, labels, title="Visualizing KMeans clusters with Euclidean distance")
\`\`\`

## Conclusion

This project provides a comprehensive example of clustering time series data and visualizing the results using various techniques. The provided utility functions and Jupyter notebook can be adapted to different datasets and clustering methods.
