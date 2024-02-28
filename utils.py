import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

red = "#962428"
yellow = "#fefdc5"
blue = "#374b99"
green = "#7db9ab"


def run_clustering_method(clustering_method, X, metric='precomputed', verbose=True):

    first_start_time = time.time()
    inertias = []
    silhouettes = []
    labels_dict = dict()

    for k in range(2, 11):
        start = time.time()
        if verbose:
            print("Number of clusters:", k)

        clustering = clustering_method(k) 

        # fit the clustering method
        clustering = clustering.fit(X)
        
        # calculate different scores for the clustering
        # silhouette score
        if hasattr(clustering, 'labels_'):
            labels = clustering.labels_
        elif hasattr(clustering, 'labels'):
            labels = clustering.labels
        else:
            labels = clustering.predict(X)
        
        silhouette_avg = silhouette_score(X, labels, metric=metric)
        if verbose:
            print("The average silhouette_score is :", silhouette_avg)

        silhouettes.append(silhouette_avg)

        # inertia
        if hasattr(clustering, 'inertia_'):
            inertia = clustering.inertia_
        elif hasattr(clustering, 'inertia'):
            inertia = clustering.inertia
        else:
            inertia = None

        if verbose:
            print("The inertia is :", inertia)
        
        if inertia is not None:
            inertias.append(inertia)

        # Y matrix
        labels_dict[k] = labels

        if verbose:
            print("Time:", time.time() - start)
            print("--------------------------------------------------")
        
    if verbose:
        print("Total time:", time.time() - first_start_time)

    return inertias, silhouettes, labels_dict


# plot the dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)


    plt.figure(figsize=(20,10))
    plt.title("Hierarchical Clustering Dendrogram")

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


def plot_inertia(inertias, title):
    # elbow method
    plt.plot(range(2, 11), inertias)
    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.show()


def plot_silhouette(silhouettes, title):
    # silhouette score (the higher the better)
    plt.plot(range(2, 11), silhouettes)
    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.show()


def plot_silhouette_analysis(X, labels, title):
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1
    ax1.set_xlim([-1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (len(np.unique(labels)) + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(len(np.unique(labels))):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        y_upper = y_lower + size_cluster_i

        cmap = plt.get_cmap("Spectral")
        color = cmap(float(i) / len(np.unique(labels)))

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title(title)
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    plt.show()


def plot_heatmap(Y, cmap=[green, red, blue, yellow]):
    sns.heatmap(Y, cmap=cmap)
    plt.show()


def visualize_clusters(X, labels, title):
    # Visualisation des séries temporelles appartenant aux mêmes classes
    plt.figure(figsize=(20, 10))
    colors = [red, blue, green, yellow]

    for yi, color in enumerate(colors):
        plt.subplot(3, 4, yi + 1)
        for i, xx in X[labels == yi].iterrows():
            plt.plot(xx.ravel(), color=color, alpha=.2)
        # Plot the mean of each cluster as a thick horizontal line knowing that cah doesn't have cluster_centers_
        cluster_mean = np.mean(X[labels == yi], axis=0)
        plt.plot(cluster_mean, color='black', linestyle='dashed', linewidth=2)
        plt.xlim(0, X.shape[1])
        plt.ylim(-4, 4)
        plt.text( 0.55, 0.85,'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
        if yi == 1:
            plt.title(title)