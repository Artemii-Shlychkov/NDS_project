import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import string
import sys
import os 
import scipy as sp
from scipy import sparse
import sklearn

## add your packages ##

import time
import pickle
#import memory_profiler
from packaging.version import parse as parse_version
from memory_profiler import profile

# Load the memory_profiler extension
#get_ipython().run_line_magic('load_ext', 'memory_profiler')

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture as GMM
from scipy import stats
import seaborn as sns
import umap
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap.umap_ as umap

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def mog_bic(
    x: np.ndarray, m: np.ndarray, S: np.ndarray, p: np.ndarray
) -> tuple[float, float]:
    """Compute the BIC for a fitted Mixture of Gaussian model

    Parameters
    ----------

    x: np.array, (n_samples, n_dims)
        Input data

    m: np.array, (n_clusters, n_dims)
        Means

    S: np.array, (n_clusters, n_dims, n_dims)
        Covariances

    p: np.array, (n_clusters, )
        Cluster weights / probabilities

    Return
    ------

    bic: float
        BIC

    LL: float
        Log Likelihood
    """
    n_clusters = len(m)  # number of clusters
    n_dims = x.shape[1]  # number of dimensions
    n_samples = x.shape[0]  # number of samples

    # Calculate the log likelihood of the data given the model
    expectation = np.zeros((n_samples, n_clusters), dtype=np.float64)

    for k in range(n_clusters):
        pi_k = p[k]
        mu_k = m[k]
        cov_k = S[k] + 1e-6 * np.eye(
            n_dims
        )  # Adding a small value for numerical stability
        expectation[:, k] = pi_k * stats.multivariate_normal(mean=mu_k, cov=cov_k).pdf(
            x
        )

    totals = np.sum(expectation, axis=1)

    # Ensure totals are not zero to avoid log(0)
    if np.any(totals == 0):
        raise ValueError("Some data points have zero probability under the model.")

    LL = np.sum(np.log(totals))

    # Number of parameters
    P = (n_clusters - 1) + n_clusters * n_dims + n_clusters * n_dims * (n_dims + 1) / 2

    bic = -2 * LL + P * np.log(n_samples)

    return bic, LL


def plot_ellipse(mean: float, cov: np.array, ax: plt.Axes, color: str, cluster_n: int) -> None:
    
    ls_list = ['-', '--', ':', '-.']

    # Convert the color to RGB
    RGB_color = list(matplotlib.colors.to_rgb(color))

    alpha_fill = 0.2
    # Add alpha to the RGB color
    RGBA_color = RGB_color + [alpha_fill]
    
    for i in range(cluster_n):
        
        v, w = np.linalg.eigh(cov[i])
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees

        ell = matplotlib.patches.Ellipse(
                xy=mean[i],
                width=v[0],
                height=v[1],
                angle=180.0 + angle,
                edgecolor='black',
                linewidth = 2.5,
                facecolor = RGB_color,  # Ensure color index does not exceed available colors
                label=f'Cluster {i}',
                
                ls = ls_list[i],
                
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        ax.add_artist(ell)

        


def get_optimal_cluster_number(possible_clusters: tuple, num_seeds: int, data: np.array, bic_mode: str = 'manual', plot: bool = False) -> int:
    """
    Get the optimal number of clusters for a Gaussian Mixture Model
    
    
    Parameters
    ----------
    
    possible_clusters: tuple
    
    num_seeds: int
    
    data: np.array, (n_samples, n_dims)
    
    bic_mode: str, default='manual'
    
    plot: bool, default=False
    
    Returns
    -------
    
    optimal_cluster_number: int
    
    """
    
    BIC = np.zeros((num_seeds, len(possible_clusters)))
    LL = np.zeros((num_seeds, len(possible_clusters)))

    for i, n_clusters in enumerate(possible_clusters):
        for seed in range(num_seeds):
            gmm = GMM(n_components=n_clusters, random_state=seed).fit(data)

            cluster_ids = gmm.predict(data)
            likelihood = gmm.score_samples(data)
            weights = gmm.weights_
            means = gmm.means_
            covariances = gmm.covariances_

            if bic_mode == 'manual':
                BIC[seed, i], LL[seed, i] = mog_bic(
                    data, means, covariances, weights
                )
            elif bic_mode == 'sklearn':
                BIC[seed, i] = gmm.bic(data)
    model_index, cluster_index = np.where(BIC == BIC.min())
    optimal_cluster_number = possible_clusters[cluster_index[0]]
    lowest_bic = BIC[model_index[0], :]
    print(f"Optimal number of clusters: {optimal_cluster_number}")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.errorbar(
        possible_clusters,
        BIC.mean(axis=0),
        yerr=BIC.std(axis=0),
        label="BIC",
        fmt="o-",
        c="black",
        )
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("BIC")
        ax.scatter(optimal_cluster_number, lowest_bic[cluster_index[0]], c="green", label="Optimal number of clusters", zorder = 10, s=20)
        ax.legend()
        ax.set_title("Optimal number of clusters for the GMM")
        ax.set_yticks([])
        ax.set_xticks(possible_clusters)

    return optimal_cluster_number

def plot_TSNE(ax: plt.Axes, data: np.ndarray, labels: np.ndarray, title: str, display_accuracy: bool = False, perplexity:int = 30) -> None:
    """
    Plot t-SNE data with colors

    Parameters
    ----------
    ax: plt.Axes
        Axis to plot on

    data: np.ndarray
        TSNE-fit data to plot

    labels: np.ndarray
        Color labels to use for plotting

    title: str
        Title of the plot

    Return
    ------
    None
    """
    if display_accuracy:
        accuracy,recall = knn_classification(data, labels)
        ax.text(0.01,0.95,f"Accuracy: {accuracy:.2f}\nRecall: {recall:.2f}\nPerplexity: {int(perplexity)}", transform=ax.transAxes, fontsize=8, verticalalignment='top')

    ax.scatter(data[:, 0], data[:, 1], s=10, c=labels, alpha=1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    ax.set_title(title)
    
    sns.despine(ax=ax, left=True, bottom=True)


def plot_dim1_dim2(
    ax: plt.Axes, data: dict, labels: np.ndarray, title: str,type: str, display_stats: bool = False, 
) -> None:
    """
    Plot PC1 and PC2 data with colors

    Parameters
    ----------
    ax: plt.Axes
        Axis to plot on

    data: dict
        dictionary containing the reduced data and the accuracy and recall scores

    labels: np.ndarray
        Color labels to use for plotting

    title: str
        Title of the plot

    Return
    ------
    None
    """

    # data is a dict 
    datapoints = data[type]['data']
    accuracy = data[type]['accuracy']
    recall = data[type]['recall']


    if display_stats:
        ax.text(0.01, 0.95, f"Accuracy: {accuracy:.2f}\nRecall: {recall:.2f}", transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.scatter(datapoints[:, 0], datapoints[:, 1], s=10, c=labels, alpha=1)
    if type == 'PCA':
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    elif type == 'LDA':
        ax.set_xlabel("LDA1")
        ax.set_ylabel("LDA2")
    elif type == 'NCA':
        ax.set_xlabel("NCA1")
        ax.set_ylabel("NCA2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis = "equal"
    
    ax.set_title(title)

def plot_umap(ax: plt.Axes, data: np.ndarray, labels: np.ndarray, title: str, display_accuracy: bool=False) -> None:
    """
    Plot UMAP data with colors

    Parameters
    ----------
    ax: plt.Axes
        Axis to plot on

    data: np.ndarray
        Data to plot

    labels: np.ndarray
        color labels to use for plotting

    title: str
        Title of the plot

    Return
    ------
    None
    """
    if display_accuracy:
        accuracy, recall = knn_classification(data, labels)
        ax.text(0.01, 0.95, f"Accuracy: {accuracy:.2f}\nRecall: {recall:.2f}", transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.scatter(data[:, 0], data[:, 1], c=labels, s=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis = "equal"
    sns.despine(ax=ax, bottom=True, left=True)
    ax.set_title(title)

def calculate_font_size(base_size: int, factor: float) -> int:
    """
    Calculate font size based on base size and factor

    Parameters
    ----------
    base_size: int
        Base size of the font

    factor: float
        Factor to scale the font size

    Return
    ------
    int
        Font size
    """

    return max(base_size * factor, 4)  # Ensure minimum font size of 6

def knn_classification(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    k: int = None,
) -> tuple:
    """
    Perform KNN classification on the data

    Parameters
    ----------
    X: np.ndarray
        Data to classify

    y: np.ndarray
        Labels of the data

    test_size: float
        Fraction of the data to use as test data

    random_state: int
        Random state for reproducibility

    Return
    ------
    tuple
        Tuple containing the accuracy and recall scores

    """
    if k is None:
        k = int(np.sqrt(X.shape[0]))
    knn = KNeighborsClassifier(n_neighbors=k)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")

    return accuracy, recall

def dim_reduction(data: np.ndarray, labels: np.ndarray) -> dict:
    """
    Perform dimensionality reduction on the data. The data is split into training and testing data. 
    The training data is used to fit the model and the testing data is used to evaluate the model. 
    The accuracy and recall scores are calculated for each dimensionality reduction technique.
    The methods used are PCA, NCA, and LDA. 

    Parameters
    ----------
    data: np.ndarray
        Data to reduce

    labels: np.ndarray
        Labels of the data

    Return
    ------
    dict
        Dictionary containing the reduced data and the accuracy and recall scores
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=int(np.sqrt(X_train.shape[0])))

    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=2, random_state=42),
    )
    pca = make_pipeline(StandardScaler(), PCA(n_components=2))

    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

    methods = {"PCA": pca, "NCA": nca, "LDA": lda}
    reduced_data = {}

    for name, model in methods.items():
        reduced_data[name] = {}

        model.fit(X_train, y_train)
        X_train_reduced = model.transform(X_train)
        X_test_reduced = model.transform(X_test)

        knn.fit(X_train_reduced, y_train)
        y_pred = knn.predict(X_test_reduced)

        accuracy = knn.score(X_test_reduced, y_test)
        recall = recall_score(y_test, y_pred, average="weighted")

        reduced_data[name]["data"] = model.transform(data)
        reduced_data[name]["accuracy"] = accuracy
        reduced_data[name]["recall"] = recall

    for key in reduced_data.keys():
        print(f"{key} accuracy: {reduced_data[key]['accuracy']:.2f}")
        print(f"{key} recall: {reduced_data[key]['recall']:.2f}")

    return reduced_data